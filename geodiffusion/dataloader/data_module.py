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

# Called once per DataLoader worker process at spawn time (multi-GPU / multi-worker).
# Each worker is a separate subprocess that inherits the parent's environment, so
# without this hook every worker would launch its own GDAL/OMP/MKL thread pools —
# causing severe CPU oversubscription on shared HPC nodes.
def _worker_init_fn(worker_id: int) -> None:
    """Limit native thread pools inside DataLoader workers to avoid thread exhaustion."""
    os.environ["GDAL_NUM_THREADS"] = "1"   # hard override (not setdefault — must take effect)
    # NOTE: do NOT add GDAL_DISABLE_READDIR_ON_OPEN here; rasterio needs directory
    # reads to find the .jpg.aux.xml sidecar that carries the image CRS/georeference.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import cv2; cv2.setNumThreads(0)    # optional dep — skip if not installed
    except Exception:
        pass
    torch.set_num_threads(1)                # torch always importable; no try/except needed


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

        cfg.data.data_root       (e.g. "/path/to/processed/dataset/root")
        cfg.data.max_gt_segments (optional, default 100)  # max segments per image after filtering
        cfg.data.max_train_samples   (optional, default None, i.e. no cap) 
        cfg.data.max_val_samples     (optional, default None, i.e. no cap)
        cfg.data.densify             (optional, default False)
        cfg.data.max_segment_length  (optional, default 0.10)  # 0.10 ≈ 51 m at 512 px / 1 m GSD
        cfg.training.batch_size
        cfg.training.num_workers
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: VectorRoadDataset | None = None
        self.val_dataset: VectorRoadDataset | None = None

    # Build train/val VectorRoadDataset objects from config for Lightning
    def setup(self, stage: str | None = None):
        d = self.cfg.data
        kwargs = dict(
            data_root=d.data_root,
            max_gt_segments=d.max_gt_segments,
            densify=bool(d.get("densify", False)),
            max_segment_length=float(d.get("max_segment_length", 0.10)),
            image_size=int(d.get("image_size", 512)),
            gsd_m=float(d.get("gsd_m", 1.0)),
            use_exclusion_csv=bool(d.get("use_exclusion_csv", False)),
            exclusion_csv_path=d.get("exclusion_csv_path", None),
        )
        if stage in ("fit", None):
            self.train_dataset = VectorRoadDataset(split="train", **kwargs)
            self.val_dataset = VectorRoadDataset(split="val", **kwargs)
            if os.environ.get("LOCAL_RANK", "0") == "0":
                self._print_dataset_summary()
        elif stage == "validate" and self.val_dataset is None:
            self.val_dataset = VectorRoadDataset(split="val", **kwargs)

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
            "CSV removed",
            str(getattr(tr, "_n_removed_exclusion_csv", 0)),
            str(getattr(va, "_n_removed_exclusion_csv", 0)),
        )
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
