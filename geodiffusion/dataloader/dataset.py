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

os.environ.setdefault("PROJ_NETWORK", "OFF")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "NO")
os.environ.setdefault("GDAL_CACHEMAX", "512")
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "25000000")

import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

import torch
from torch.utils.data import Dataset

from rich.console import Console

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
        densify: bool = True,
        max_segment_length: float = 0.1,
        image_size: int = 512,
        gsd_m: float = 1.0, # ground sampling distance in meters (used to convert max_segment_length from normalized [0,1] to pixel units)
        use_exclusion_csv: bool = False,
        exclusion_csv_path: str | None = None,
    ):
        self.data_root = data_root
        self.max_gt_segments = int(max_gt_segments)
        self.densify = bool(densify)
        self.max_segment_length = float(max_segment_length)
        self.image_size = int(image_size)
        self.gsd_m = float(gsd_m)
        self.split = split
        self.use_exclusion_csv = use_exclusion_csv
        self.exclusion_csv_path = exclusion_csv_path

        # ── locate split directory ────────────────────────────────────────────
        base = os.path.join(data_root, split)

        # ── index files ──────────────────────────────────────────────────────
        # Each satellite image <id>_sat.jpg is paired with <id>_osm.geojson in the same dir.
        paths = sorted(Path(base).glob("*_sat.jpg"))
        self.sat_images = [str(p) for p in paths]
        self.road_files = [str(p.with_name(p.stem.removesuffix("_sat") + "_osm.geojson")) for p in paths]
        self.sample_ids = [p.stem.removesuffix("_sat") for p in paths]

        # ── filter counters ───────────────────────────────────────────────────
        self._n_removed_exclusion_csv = 0

        # ── apply filters ─────────────────────────────────────────────────────
        if self.use_exclusion_csv:
            self._apply_exclusion_csv_filter()

        # ── summary fields (used by data_module table) ────────────────────────
        self._base = base
        self._seg_m = max_segment_length * image_size * gsd_m

    # ──────────────────────────────────────────────────────────────────────────
    # Filtering helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_exclusion_csv_filter(self) -> None:
        """Remove samples listed in an exclusions CSV by crop_id/sample_id (applied to both train and val)."""
        p = Path(self.exclusion_csv_path)

        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            key = next((k for k in ("crop_id", "sample_id") if k in (reader.fieldnames or [])), None)
            assert key is not None, f"Exclusion CSV has no crop_id/sample_id column: {p}"
            excluded_ids = {(row.get(key) or "").strip() for row in reader}

        keep = [(a, b, c) for a, b, c in zip(self.sat_images, self.road_files, self.sample_ids)
                if c not in excluded_ids and not any(c.startswith(e + "_") for e in excluded_ids)]
        self._n_removed_exclusion_csv += len(self.sample_ids) - len(keep)
        self.sat_images, self.road_files, self.sample_ids = map(list, zip(*keep)) if keep else ([], [], [])

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset protocol
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sat_images)

    def __getitem__(self, index: int) -> dict:
        with rasterio.open(self.sat_images[index]) as ds:
            data = ds.read()
            segments = self._parse_roads(self.road_files[index], ds, data.shape[-1])
        assert data.shape[0] >= 3, f"Expected ≥3-channel image, got {data.shape[0]}"
        # Drop samples with no road annotations or more segments than the tensor
        # can hold — truncating would corrupt the GT (incomplete annotation is
        # worse than no annotation for matching-based losses).
        if not segments or len(segments) > self.max_gt_segments:
            effective_index = -1
        else:
            effective_index = index
        road_data, invalid_mask = self._pad_or_trim(segments if effective_index != -1 else [])
        return {
            "image":        torch.from_numpy(data[:3].copy()),
            "road_data":    torch.tensor(road_data,    dtype=torch.float32),
            "invalid_mask": torch.tensor(invalid_mask, dtype=torch.bool),
            "index":        effective_index,
        }

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
        # Drop degenerate and artifact segments. Normalised length > √2 (~1.414)
        # means the segment spans more than the image half-diagonal — these are
        # reprojection artifacts whose clipped endpoints are not real road termini.
        # Also drop zero-length segments that would produce NaN velocities.
        norm = [
            seg for seg in norm
            if (seg[2] - seg[0]) ** 2 + (seg[3] - seg[1]) ** 2 > 1e-8
            and (seg[2] - seg[0]) ** 2 + (seg[3] - seg[1]) ** 2 <= 2.0  # ≤ √2² = 2
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

    # ──────────────────────────────────────────────────────────────────────────
    # Padding Segments to max_gt_segments with invalid_mask
    # ──────────────────────────────────────────────────────────────────────────
    def _pad_or_trim(
        self, segments: list[tuple[float, float, float, float]]
    ) -> tuple[list, list]:
        """Pad segments to max_gt_segments with out-of-range sentinels.

        Callers are responsible for ensuring len(segments) <= max_gt_segments
        before calling this method (oversized samples should be dropped at the
        __getitem__ level so the batch never sees a truncated annotation).

        Returns:
            (segments_padded, invalid_mask)  — both length max_gt_segments
        """
        n = self.max_gt_segments
        assert len(segments) <= n, (
            f"_pad_or_trim called with {len(segments)} segments > max_gt_segments={n}. "
            "Filter the sample before calling."
        )
        valid_count = len(segments)
        padded = list(segments)
        while len(padded) < n:
            padded.append((2.0, 2.0, 2.0, 2.0))  # out-of-range sentinel; valid coords are in [-1, 1]
        invalid_mask = [False] * valid_count + [True] * (n - valid_count)
        return padded, invalid_mask
