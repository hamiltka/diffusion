"""
Lightning DataModule for vector road-segment flow-matching training.
"""
import os

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, default_collate

from geodiffusion.dataloader.dataset import VectorRoadDataset


def _worker_init_fn(worker_id: int) -> None:
    """Limit native thread pools inside DataLoader workers to avoid thread exhaustion."""
    # Keep each worker process single-threaded to avoid oversubscribing CPU cores.
    os.environ["GDAL_NUM_THREADS"] = "1"
    # NOTE: Do NOT set GDAL_DISABLE_READDIR_ON_OPEN here — the USGS JPEG
    # satellite images store their CRS/georeference in .jpg.aux.xml sidecar
    # files.  Disabling directory reads prevents GDAL from discovering those
    # sidecars, causing rasterio to open images without georeference info and
    # producing zero valid road segments from every sample.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _collate(batch):
    """Drop None / error samples so batch size stays consistent."""
    # Dataset may return None or index=-1 for unusable samples; remove them here.
    batch = [b for b in batch if b is not None and b.get("index", 0) != -1]
    if not batch:
        # Lightning step can check this marker and skip forward/backward safely.
        return {"_skip_batch": True}
    return default_collate(batch)


class VectorDiffusionDataModule(pl.LightningDataModule):
    """
    DataModule wrapping :class:`VectorRoadDataset` for train / val splits.

    Config keys used (new ``data`` namespace)::

        cfg.data.data_root
        cfg.data.max_gt_segments
        cfg.data.max_train_samples   (optional)
        cfg.data.max_val_samples     (optional)
        cfg.data.densify             (optional, default True)
        cfg.data.max_segment_length  (optional, default 0.10)
        cfg.training.batch_size
        cfg.training.num_workers
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: VectorRoadDataset | None = None
        self.val_dataset: VectorRoadDataset | None = None

    def setup(self, stage: str | None = None):
        d = self.cfg.data
        # Shared dataset construction kwargs for both train/val splits.
        # These are all data-quality and geometry-normalization controls.
        kwargs = dict(
            data_root=d.data_root,
            max_gt_segments=d.max_gt_segments,
            densify=bool(d.get("densify", True)),
            max_segment_length=float(d.get("max_segment_length", 0.10)),
            image_size=int(d.get("image_size", 512)),
            gsd_m=float(d.get("gsd_m", 1.0)),
            min_valid_features_train=int(d.get("min_valid_features_train", 0)),
            min_valid_features_val=int(d.get("min_valid_features_val", 0)),
            max_valid_features_train=int(d.get("max_valid_features_train", 0)),
            max_valid_features_val=int(d.get("max_valid_features_val", 0)),
            filter_nodata=bool(d.get("filter_nodata", False)),
            nodata_threshold=float(d.get("nodata_threshold", 0.30)),
            nodata_white_threshold=int(d.get("nodata_white_threshold", 250)),
            nodata_black_threshold=int(d.get("nodata_black_threshold", 5)),
            use_exclusion_csv=bool(d.get("use_exclusion_csv", False)),
            exclusion_csv_path=d.get("exclusion_csv_path", None),
            use_source_blocklist=bool(d.get("use_source_blocklist", False)),
            blocklist_path=d.get("blocklist_path", None),
            augment=d.get("augment", None),  # None = auto (train=True, val=False)
        )
        if stage in ("fit", None):
            # Build train split with optional cap for quick experiments.
            self.train_dataset = VectorRoadDataset(
                split="train",
                max_samples=d.get("max_train_samples", None),
                **kwargs,
            )
            # Build val split separately so filtering stats are measured per split.
            self.val_dataset = VectorRoadDataset(
                split="val",
                max_samples=d.get("max_val_samples", None),
                **kwargs,
            )
            # Avoid duplicate table output from all DDP ranks.
            if os.environ.get("LOCAL_RANK", "0") == "0":
                self._print_dataset_summary()
        elif stage == "validate" and self.val_dataset is None:
            # Support validate-only entrypoints that never called fit().
            self.val_dataset = VectorRoadDataset(
                split="val",
                max_samples=d.get("max_val_samples", None),
                **kwargs,
            )

    def _print_dataset_summary(self) -> None:
        # Convenience aliases for a compact table definition.
        tr = self.train_dataset
        va = self.val_dataset
        tbl = Table(
            title="[bold white]Dataset Summary[/]",
            show_header=True,
            header_style="bold bright_cyan",
            box=None,
            padding=(0, 2),
            title_style="bold bright_cyan",
        )
        tbl.add_column("", style="bold bright_cyan", no_wrap=True)
        tbl.add_column("Train", style="white")
        tbl.add_column("Val",   style="white")
        # Values below come from dataset internals populated during indexing/filtering.
        tbl.add_row("Source",          tr._base, va._base)
        tbl.add_row("Images",          str(len(tr)), str(len(va)))
        tbl.add_row("Max seg length",  f"~{tr._seg_m:.0f} m", "")
        tbl.add_row("Road type filter", "19 driveable types", "")
        tbl.add_row(
            "Min valid features",
            str(getattr(tr, "min_valid_features_train", 0)),
            str(getattr(va, "min_valid_features_val", 0)),
        )
        tbl.add_row(
            "Max valid features",
            str(getattr(tr, "max_valid_features_train", 0)),
            str(getattr(va, "max_valid_features_val", 0)),
        )
        tbl.add_row(
            "Sparse removed",
            str(getattr(tr, "_n_removed_sparse", 0)),
            str(getattr(va, "_n_removed_sparse", 0)),
        )
        tbl.add_row(
            "Dense removed",
            str(getattr(tr, "_n_removed_dense", 0)),
            str(getattr(va, "_n_removed_dense", 0)),
        )
        tbl.add_row(
            "No-data removed",
            str(getattr(tr, "_n_removed_nodata", 0)),
            str(getattr(va, "_n_removed_nodata", 0)),
        )
        tbl.add_row(
            "CSV removed",
            str(getattr(tr, "_n_removed_exclusion_csv", 0)),
            str(getattr(va, "_n_removed_exclusion_csv", 0)),
        )
        tbl.add_row(
            "Blocklist removed",
            str(getattr(tr, "_n_removed_blocklist", 0)),
            str(getattr(va, "_n_removed_blocklist", 0)),
        )
        # Augmentations are only active in train split by dataset design.
        tbl.add_row("Augmentations",   "hflip  vflip  rot90  colorjitter  occlusion", "[dim]disabled[/]")
        Console().print(tbl)

    def _loader(self, dataset: VectorRoadDataset, shuffle: bool) -> DataLoader:
        nw = self.cfg.training.get("num_workers", 0)
        return DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=shuffle,
            num_workers=nw,
            collate_fn=_collate,
            worker_init_fn=_worker_init_fn if nw > 0 else None,
            # Useful for host->GPU transfer latency; safe even when running on CPU.
            pin_memory=True,
            # Keep fixed batch shapes for DDP and stable optimization dynamics.
            drop_last=True,
            # Reuse worker processes across epochs to reduce startup overhead.
            persistent_workers=nw > 0,
            # forkserver avoids some deadlocks/resource inheritance issues vs fork.
            multiprocessing_context="forkserver" if nw > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)
