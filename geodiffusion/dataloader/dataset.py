"""
Dataset for satellite image + GeoJSON road-network pairs.

Each sample:
    image:        uint8 tensor [3, H, W]
    road_data:    float32 tensor [max_gt_segments, 4]   (x1,y1,x2,y2) in [-1,1]
    invalid_mask: bool tensor [max_gt_segments]          True = padded / invalid
    index:        int

Data directory layout::

    <data_root>/
      train/
        <id>_sat.jpg
        <id>_roads.geojson
      val/
        ...
      test/
        ...

Augmentation (train split only):
    - Random horizontal flip  (p=0.5)  — mirrors x coords
    - Random vertical flip    (p=0.5)  — mirrors y coords
    - Random 90° rotation     (p=0.5)  — rotates image + segment coords

Road type filtering:
    Only road features whose ``highway`` property is in VALID_HIGHWAY_TYPES
    are included.  Features with no highway property are kept (permissive).
"""
from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import math
import numpy as np

os.environ.setdefault("PROJ_NETWORK", "OFF")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "NO")
os.environ.setdefault("GDAL_CACHEMAX", "512")
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "25000000")

import rasterio
import rasterio.crs
from rasterio.crs import CRS
import rasterio.warp
from rasterio.warp import transform as warp_transform

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.transforms import ColorJitter

from rich.console import Console
from rich.table import Table
from rich import box as rich_box

from geodiffusion.preprocessing.densify import densify_segments

VALID_HIGHWAY_TYPES: set[str] = frozenset({
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    "living_street", "road", "busway", "bus_guideway",
    "raceway", "escape", "construction",
})


class VectorRoadDataset(Dataset):

    def __init__(
        self,
        split: str,
        data_root: str | None = None,
        max_gt_segments: int = 1000,
        max_samples: int | None = None,
        densify: bool = True,
        max_segment_length: float = 0.1,
        augment: bool | None = None,
        image_size: int = 512,
        gsd_m: float = 1.0,
        min_valid_features_train: int = 0,
        min_valid_features_val: int = 0,
        max_valid_features_train: int = 0,
        max_valid_features_val: int = 0,
        filter_nodata: bool = False,
        nodata_threshold: float = 0.3,
        nodata_white_threshold: int = 250,
        nodata_black_threshold: int = 5,
        use_exclusion_csv: bool = False,
        exclusion_csv_path: str | None = None,
        use_source_blocklist: bool = False,
        blocklist_path: str | None = None,
    ):
        self.data_root = data_root
        self.max_gt_segments = int(max_gt_segments)
        self.densify = bool(densify)
        self.max_segment_length = float(max_segment_length)
        self.image_size = int(image_size)
        self.gsd_m = float(gsd_m)
        self.split = split
        self.min_valid_features_train = int(min_valid_features_train)
        self.min_valid_features_val = int(min_valid_features_val)
        self.max_valid_features_train = int(max_valid_features_train)
        self.max_valid_features_val = int(max_valid_features_val)
        self.filter_nodata = bool(filter_nodata)
        self.nodata_threshold = float(nodata_threshold)
        self.nodata_white_threshold = nodata_white_threshold
        self.nodata_black_threshold = nodata_black_threshold
        self.use_exclusion_csv = use_exclusion_csv
        self.exclusion_csv_path = exclusion_csv_path
        self.use_source_blocklist = use_source_blocklist
        self.blocklist_path = blocklist_path
        self.augment = (split == "train") if augment is None else bool(augment)

        # ── locate split directory ────────────────────────────────────────────
        base = os.path.join(data_root, split)
        if not os.path.exists(base):
            candidate = os.path.join(data_root, "train")
            print(f"[VectorRoadDataset] split dir '{base}' not found, using {candidate}")
            base = candidate

        # ── index files ──────────────────────────────────────────────────────
        glob = sorted(Path(base).glob("*_sat.jpg"))
        self.sat_images = [str(f) for f in glob]
        self.road_files = [f.replace("_sat.jpg", "_osm.geojson") for f in self.sat_images]
        self.sample_ids = [os.path.basename(p).replace("_sat.jpg", "") for p in self.sat_images]

        # ── filter counters ───────────────────────────────────────────────────
        self._n_removed_blocklist = 0
        self._n_removed_sparse = 0
        self._n_removed_dense = 0
        self._n_removed_nodata = 0
        self._n_removed_exclusion_csv = 0

        # ── apply filters ─────────────────────────────────────────────────────
        self._apply_exclusion_csv_filter()
        self._apply_source_blocklist()
        self._apply_feature_density_filter()
        self._apply_nodata_filter()

        # ── cap to max_samples (reproducible random subset) ───────────────────
        if max_samples is not None and len(self.sat_images) > max_samples:
            rng = random.Random(42)
            indices = sorted(rng.sample(range(len(self.sat_images)), max_samples))
            self.sat_images = [self.sat_images[i] for i in indices]
            self.road_files = [self.road_files[i] for i in indices]
            self.sample_ids = [self.sample_ids[i] for i in indices]

        # ── summary fields (used by data_module table) ────────────────────────
        self._base = base
        self._seg_m = max_segment_length * image_size * gsd_m

    # ──────────────────────────────────────────────────────────────────────────
    # Filtering helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _source_from_sample_id(sample_id: str) -> str:
        """Extract source tile id from '<source>_crop_xxxx' style sample ids."""
        return sample_id.split("_crop_")[0]

    def _apply_source_blocklist(self) -> None:
        """Optionally remove samples whose source tile appears in a CSV blocklist."""
        if not self.use_source_blocklist:
            return
        p = Path(self.blocklist_path)
        if not p.exists():
            Console().print(f"[yellow]Blocklist not found:[/] {p}. Continuing without blocklist.")
            return
        blocked = set()
        with open(p) as f:
            header = f.readline().strip().split(",")
            try:
                source_idx = header.index("source")
            except ValueError:
                Console().print(
                    f"[yellow]Blocklist missing 'source' column:[/] {p}. Skipping."
                )
                return
            for line in f:
                cols = line.strip().split(",")
                if len(cols) > source_idx:
                    blocked.add(cols[source_idx])
        keep_idx = [
            i for i, sid in enumerate(self.sample_ids)
            if self._source_from_sample_id(sid) not in blocked
        ]
        self._n_removed_blocklist += len(self.sample_ids) - len(keep_idx)
        self.sat_images = [self.sat_images[i] for i in keep_idx]
        self.road_files = [self.road_files[i] for i in keep_idx]
        self.sample_ids = [self.sample_ids[i] for i in keep_idx]

    def _apply_exclusion_csv_filter(self) -> None:
        """Remove samples listed in an exclusions CSV by crop_id/sample_id (applied to both train and val)."""
        if not self.use_exclusion_csv:
            return
        p = Path(self.exclusion_csv_path)
        if not p.exists():
            Console().print(
                f"[yellow]Exclusion CSV not found:[/] {p}. Continuing without exclusion CSV."
            )
            return
        _STRIP_SUFFIXES = ("_NAIP", "_naip")

        def _normalise(sid: str) -> str:
            for sfx in _STRIP_SUFFIXES:
                if sid.endswith(sfx):
                    sid = sid[: -len(sfx)]
            return sid

        excluded_ids: set[str] = set()
        try:
            with open(p, newline="") as f:
                reader = csv.DictReader(f)
                key = None
                for k in ("crop_id", "sample_id"):
                    if k in (reader.fieldnames or []):
                        key = k
                        break
                if key is None:
                    Console().print(
                        f"[yellow]Exclusion CSV missing crop_id/sample_id column:[/] {p}. Skipping."
                    )
                    return
                for row in reader:
                    sid = (row.get(key) or "").strip()
                    excluded_ids.add(_normalise(sid))
        except Exception as e:
            Console().print(
                f"[yellow]Failed reading exclusion CSV:[/] {p} ({e}). Skipping."
            )
            return
        keep_idx = [
            i for i, sid in enumerate(self.sample_ids)
            if _normalise(sid) not in excluded_ids
        ]
        self._n_removed_exclusion_csv += len(self.sample_ids) - len(keep_idx)
        self.sat_images = [self.sat_images[i] for i in keep_idx]
        self.road_files = [self.road_files[i] for i in keep_idx]
        self.sample_ids = [self.sample_ids[i] for i in keep_idx]

    @staticmethod
    def _count_valid_road_features(road_path: str) -> int:
        """Fast pre-filter count of valid road features in a GeoJSON file."""
        if not os.path.exists(road_path):
            return 0
        try:
            with open(road_path) as f:
                road_data = json.load(f)
        except Exception:
            return 0
        n = 0
        for feature in road_data.get("features", []):
            props = feature.get("properties", {})
            hw = props.get("highway", "")
            if hw and hw not in VALID_HIGHWAY_TYPES:
                continue
            geom = feature.get("geometry", None)
            if geom is None:
                continue
            if geom.get("type") not in {"LineString", "MultiLineString", "Polygon", "MultiPolygon"}:
                continue
            n += 1
        return n

    def _apply_feature_density_filter(self) -> None:
        """Filter sparse/dense samples using valid road-feature counts."""
        if self.split == "train":
            min_features = self.min_valid_features_train
            max_features = self.max_valid_features_train
        else:
            min_features = self.min_valid_features_val
            max_features = self.max_valid_features_val
        if min_features == 0 and max_features == 0:
            return
        keep_idx = []
        n_sparse = 0
        n_dense = 0
        for i, road_path in enumerate(self.road_files):
            n = self._count_valid_road_features(road_path)
            remove = False
            if min_features > 0 and n < min_features:
                n_sparse += 1
                remove = True
            if max_features > 0 and n > max_features:
                n_dense += 1
                remove = True
            if not remove:
                keep_idx.append(i)
        self._n_removed_sparse += n_sparse
        self._n_removed_dense += n_dense
        self.sat_images = [self.sat_images[i] for i in keep_idx]
        self.road_files = [self.road_files[i] for i in keep_idx]
        self.sample_ids = [self.sample_ids[i] for i in keep_idx]

    def _apply_nodata_filter(self) -> None:
        """Remove images with too many near-white/near-black pixels."""
        if not self.filter_nodata:
            return
        threshold = self.nodata_threshold
        keep_idx = []
        removed = 0
        for i, sat_path in enumerate(self.sat_images):
            try:
                with rasterio.open(sat_path) as ds:
                    data = ds.read()                        # [C, H, W]
                img = np.transpose(data, (1, 2, 0))         # [H, W, C]
                near_white = np.all(img > self.nodata_white_threshold, axis=-1)
                near_black = np.all(img < self.nodata_black_threshold, axis=-1)
                nodata_frac = float(np.sum(near_white | near_black)) / max(
                    img.shape[0] * img.shape[1], 1
                )
                if nodata_frac <= threshold:
                    keep_idx.append(i)
                else:
                    removed += 1
            except Exception:
                keep_idx.append(i)
        self._n_removed_nodata += removed
        self.sat_images = [self.sat_images[i] for i in keep_idx]
        self.road_files = [self.road_files[i] for i in keep_idx]
        self.sample_ids = [self.sample_ids[i] for i in keep_idx]

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset protocol
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sat_images)

    def __getitem__(self, index: int) -> dict:
        try:
            return self._load(index)
        except Exception as e:
            Console().print(f"[red]VectorRoadDataset error loading index {index}: {e}[/]")
            return self._dummy()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load(self, index: int) -> dict:
        sat_path = self.sat_images[index]
        road_path = self.road_files[index]
        with rasterio.open(sat_path) as ds:
            data = ds.read()                                # [C, H, W]
            img_size = data.shape[-1]
            segments = self._parse_roads(road_path, ds, img_size)
        if data.shape[0] < 3:
            raise ValueError(f"Expected ≥3-channel image, got {data.shape[0]}")
        img_tensor = torch.from_numpy(data[:3].copy())      # [3, H, W] uint8
        if self.augment and len(segments) > 0:
            img_tensor, segments = self._augment(img_tensor, segments)
        road_data, invalid_mask = self._pad_or_trim(segments)
        return {
            "image":        img_tensor,
            "road_data":    torch.tensor(road_data,    dtype=torch.float32),
            "invalid_mask": torch.tensor(invalid_mask, dtype=torch.bool),
            "index":        index,
        }

    def _augment(self, img: torch.Tensor, segs: list[tuple]) -> tuple[torch.Tensor, list[tuple]]:
        """Apply consistent augmentation to image + segment coords.

        Spatial transforms affect both image and coords.
        Colour/occlusion transforms affect only the image — GT coords are
        unchanged because the model must learn to predict them regardless of
        how the image looks.
        """
        # Horizontal flip
        if random.random() < 0.5:
            img = torch.flip(img, dims=(2,))
            segs = [(-x1, y1, -x2, y2) for x1, y1, x2, y2 in segs]
        # Vertical flip
        if random.random() < 0.5:
            img = torch.flip(img, dims=(1,))
            segs = [(x1, -y1, x2, -y2) for x1, y1, x2, y2 in segs]
        # Random 90° rotation (k ∈ {1,2,3})
        if random.random() < 0.5:
            k = random.randint(1, 3)
            img = torch.rot90(img, k=k, dims=(1, 2))
            for _ in range(k):
                segs = [(-y1, x1, -y2, x2) for x1, y1, x2, y2 in segs]
        # Clamp coords after spatial transforms
        segs = [
            (max(-1.0, min(1.0, x1)), max(-1.0, min(1.0, y1)),
             max(-1.0, min(1.0, x2)), max(-1.0, min(1.0, y2)))
            for x1, y1, x2, y2 in segs
        ]
        # Colour jitter (image only)
        jitter = ColorJitter(brightness=0.6, contrast=0.2, saturation=0.3, hue=0.1)
        pil = to_pil_image(img)
        img = (to_tensor(jitter(pil)) * 255).byte()
        # Random occlusion patch (image only)
        img = self._apply_occlusion(img)
        return img, segs

    @staticmethod
    def _apply_occlusion(img: torch.Tensor) -> torch.Tensor:
        """Paint a mean-colour rectangle over ~25% of the image (in-place copy)."""
        _, H, W = img.shape
        oh = max(1, int(H * 0.25))
        ow = max(1, int(W * 0.25))
        y0 = random.randint(0, H - oh)
        x0 = random.randint(0, W - ow)
        mean_col = img.clone().float().mean(dim=(1, 2), keepdim=True).byte()
        img = img.clone()
        img[:, y0:y0 + oh, x0:x0 + ow] = mean_col
        return img

    def _parse_roads(
        self, road_path: str, ds, img_size: int
    ) -> list[tuple[float, float, float, float]]:
        """Parse roads GeoJSON into normalised (x1,y1,x2,y2) segments."""
        if not os.path.exists(road_path):
            return []
        with open(road_path) as f:
            road_data = json.load(f)
        raw_pixel: list[tuple[float, float, float, float]] = []
        half = img_size / 2.0
        for feature in road_data.get("features", []):
            props = feature.get("properties", {})
            hw = props.get("highway", "")
            if hw and hw not in VALID_HIGHWAY_TYPES:
                continue
            geom = feature.get("geometry", None)
            if geom is None:
                continue
            gtype = geom.get("type")
            coords = geom.get("coordinates", [])
            if gtype == "LineString":
                raw_pixel.extend(self._linestring_segments(coords, ds))
            elif gtype == "MultiLineString":
                for line in coords:
                    raw_pixel.extend(self._linestring_segments(line, ds))
            elif gtype == "Polygon":
                for ring in coords:
                    raw_pixel.extend(self._linestring_segments(ring, ds))
            elif gtype == "MultiPolygon":
                for poly in coords:
                    for ring in poly:
                        raw_pixel.extend(self._linestring_segments(ring, ds))
        # Normalise pixel coords → [-1, 1]
        norm = [
            (max(-1.0, min(1.0, (x1 - half) / half)),
             max(-1.0, min(1.0, (y1 - half) / half)),
             max(-1.0, min(1.0, (x2 - half) / half)),
             max(-1.0, min(1.0, (y2 - half) / half)))
            for x1, y1, x2, y2 in raw_pixel
        ]
        # Optionally densify (operates in normalised space)
        if self.densify:
            norm = densify_segments(norm, max_length=self.max_segment_length)
        return norm

    @staticmethod
    def _linestring_segments(coords, ds) -> list[tuple[float, float, float, float]]:
        """Convert a polyline's geographic coords to (row, col) segment tuples.

        GeoJSON coordinates are always WGS84 (EPSG:4326).  The satellite images
        are stored in a projected CRS (e.g. UTM), so we must reproject before
        calling ds.index() to get pixel row/col values.
        """
        crs = CRS.from_epsg(4326)
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        xs, ys = warp_transform(crs, ds.crs, lons, lats)
        pts = [ds.index(x, y) for x, y in zip(xs, ys)]
        # pts[i] = (row, col); return as (x=col, y=row) float pixel coords
        return [
            (float(pts[i][1]), float(pts[i][0]), float(pts[i + 1][1]), float(pts[i + 1][0]))
            for i in range(len(pts) - 1)
        ]

    def _pad_or_trim(
        self, segments: list[tuple[float, float, float, float]]
    ) -> tuple[list, list]:
        """Trim to max_gt_segments and pad with zero-entries.

        Returns:
            (segments_padded, invalid_mask)  — both length max_gt_segments
        """
        n = self.max_gt_segments
        if len(segments) > n:
            random.shuffle(segments)
            segments = segments[:n]
        valid_count = len(segments)
        while len(segments) < n:
            segments.append((0.0, 0.0, 0.0, 0.0))
        invalid_mask = [False] * valid_count + [True] * (n - valid_count)
        return segments, invalid_mask

    def _dummy(self) -> dict:
        n = self.max_gt_segments
        return {
            "image":        torch.zeros(3, 512, 512, dtype=torch.uint8),
            "road_data":    torch.zeros(n, 4,        dtype=torch.float32),
            "invalid_mask": torch.ones(n,            dtype=torch.bool),
            "index":        -1,
        }
