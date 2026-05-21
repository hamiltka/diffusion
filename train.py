"""
GEODiffusion — main training entry point.

Run with Hydra overrides:
    python train.py experiment=transformer
    python train.py experiment=transformer trainer.batch_size=8
    python train.py experiment=transformer lightning.devices=[0,1,2]
    python train.py experiment=transformer 'checkpoint.checkpoint_path="runs/checkpoints/..."'
"""
import logging
import os
import sys
import warnings

# ── suppress noisy output before any framework imports ────────────────────────
warnings.filterwarnings("ignore")
for _log in [
    "pytorch_lightning", "lightning_fabric", "lightning.pytorch",
    "pytorch_lightning.utilities.rank_zero",
    "lightning.fabric.utilities.rank_zero",
    "lightning.fabric.utilities.seed",
    "lightning.fabric.utilities.distributed",
    "lightning_fabric.utilities.distributed",
    "torch.distributed",
    "lightning.pytorch.accelerators.cuda",
    "lightning.pytorch.utilities.rank_zero",
]:
    logging.getLogger(_log).setLevel(logging.ERROR)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# PyTorch 2.6+ changed weights_only default — patch before Lightning loads
import torch
torch.set_float32_matmul_precision("medium")  # silence Tensor Cores warning
_orig_load = torch.load
def _patched_load(f, *args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(f, *args, **kwargs)
torch.load = _patched_load

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
# Silence LOCAL_RANK: N — CUDA_VISIBLE_DEVICES: [...] lines
import pytorch_lightning.utilities.rank_zero as _rz
_rz.rank_zero_info = lambda *a, **k: None
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# ── ensure this directory is on sys.path ──────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from geodiffusion.lightning.lightning_module import VectorFlowLightningModule
from geodiffusion.dataloader.data_module import VectorDiffusionDataModule

console = Console(file=sys.stdout, force_terminal=True)


# ─────────────────────────────────────────────────────────────────────────────
# Rich UI helpers  (mirrors maptrace train_lightning.py)
# ─────────────────────────────────────────────────────────────────────────────

class _CompactModelSummary(RichModelSummary):
    """Model summary table only — no redundant param-count footer."""

    @staticmethod
    def summarize(
        summary_data,
        total_parameters,
        trainable_parameters,
        model_size,
        total_training_modes,
        **kw,
    ) -> None:
        from rich import get_console
        from rich.table import Table as RichTable

        con = get_console()
        col_names = list(zip(*summary_data))[0]
        tbl = RichTable(header_style="bold magenta")
        tbl.add_column(" ", style="dim")
        tbl.add_column("Name", justify="left", no_wrap=True)
        tbl.add_column("Type")
        tbl.add_column("Params", justify="right")
        if "Params per Device" in col_names:
            tbl.add_column("Params per Device", justify="right")
        tbl.add_column("Mode")
        for col in ["In sizes", "Out sizes"]:
            if col in col_names:
                tbl.add_column(col, justify="right", style="white")
        for row in list(zip(*(arr[1] for arr in summary_data))):
            tbl.add_row(*row)
        con.print(tbl)   # table only — no footer


class _EpochSummaryCallback(Callback):
    """Print one compact metric line per epoch (mirrors maptrace EpochSummaryCallback)."""

    TRAIN_KEYS = [
        (("Loss/train",),  "train_loss", ".4f"),
        (("Train/lr",),    "lr",         ".2e"),
    ]
    VAL_KEYS = [
        (("Loss/val",),   "val_loss",  ".4f"),
        (("Val/F1_05",),  "val_f1",    ".4f"),
    ]

    def __init__(self):
        super().__init__()
        self._train_snap: dict = {}

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        self._train_snap = {
            key: m[key]
            for keys, _, _ in self.TRAIN_KEYS
            for key in keys
            if key in m
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.global_rank != 0:
            return
        m = trainer.callback_metrics
        epoch = trainer.current_epoch
        total = trainer.max_epochs - 1
        parts = []
        for keys, label, fmt in self.TRAIN_KEYS:
            val = next((self._train_snap.get(k) for k in keys if k in self._train_snap), None)
            if val is not None:
                parts.append(f"[bold bright_cyan]{label}[/] [white]{val:{fmt}}[/]")
        for keys, label, fmt in self.VAL_KEYS:
            val = next((m.get(k) for k in keys if k in m), None)
            if val is not None:
                parts.append(f"[bold bright_cyan]{label}[/] [white]{val:{fmt}}[/]")
        if parts:
            bar = "[dim white]│[/]"
            console.print(
                f"[dim]Epoch [bold white]{epoch:>3}[/][dim]/{total}[/]  "
                + f"  {bar}  ".join(parts)
            )


# ─────────────────────────────────────────────────────────────────────────────
# Logger helpers
# ─────────────────────────────────────────────────────────────────────────────

class _CleanTBLogger(TensorBoardLogger):
    """Suppress Lightning's auto-logged 'epoch' scalar and hparams table."""

    def __init__(self, *args, **kwargs):
        kwargs["default_hp_metric"] = False
        super().__init__(*args, **kwargs)

    def log_metrics(self, metrics: dict, step=None):
        metrics = {k: v for k, v in metrics.items() if k != "epoch"}
        if metrics:
            super().log_metrics(metrics, step)

    def log_hyperparams(self, params, metrics=None):
        pass  # hparams already saved in Hydra's .hydra/ folder


# ─────────────────────────────────────────────────────────────────────────────
# Directory setup
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dirs(cfg: DictConfig) -> None:
    for d in [
        cfg.paths.checkpoints_folder,
        cfg.paths.logs_folder,
        f"{cfg.paths.checkpoints_folder}/{cfg.experiment_run_id}",
        f"{cfg.paths.logs_folder}/{cfg.experiment_run_id}",
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Callback setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_callbacks(cfg: DictConfig) -> tuple[list, ModelCheckpoint]:
    cb_cfg = cfg.lightning.callbacks
    ckpt_dir = f"{cfg.paths.checkpoints_folder}/{cfg.experiment_run_id}"

    best_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=cb_cfg.checkpoint_monitor,
        mode=cb_cfg.checkpoint_mode,
        save_top_k=cb_cfg.checkpoint_save_top_k,
        save_last=cb_cfg.checkpoint_save_last,
        every_n_epochs=cb_cfg.checkpoint_every_n_epochs,
        filename="epoch={epoch:02d}-val_loss={Loss/val:.4f}",
        auto_insert_metric_name=False,
    )

    periodic_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=-1,
        every_n_epochs=5,
        filename="periodic-epochepoch={epoch:02d}",
        auto_insert_metric_name=False,
    )

    return [best_ckpt, periodic_ckpt], best_ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: DictConfig) -> None:
    # In DDP, only rank 0 should print rich status panels to avoid duplicated output.
    is_rank0 = os.environ.get("LOCAL_RANK", "0") == "0"

    if is_rank0:
        # Total anchor count depends on mode: gt_seeded uses max_gt_segments, grid uses grid²×spokes.
        anchor_mode = str(cfg.anchors.get("mode", "grid"))
        if anchor_mode == "gt_seeded":
            n_anchors = cfg.data.max_gt_segments
        else:
            n_anchors = cfg.anchors.grid_size ** 2 * cfg.anchors.n_spokes

        # Build a compact run summary panel so each launch shows the effective config at a glance.
        t = Table.grid(padding=(0, 3))
        t.add_column(style="bold bright_cyan", no_wrap=True)
        t.add_column(style="white")
        t.add_row("Experiment",    cfg.get('name', 'flow_matching'))
        t.add_row("Dataset",       cfg.data.data_root)
        t.add_row("Epochs",        str(cfg.training.num_epochs))
        t.add_row("Batch size",    str(cfg.training.batch_size))
        t.add_row("N anchors",     str(n_anchors))
        t.add_row("Max GT segs",   str(cfg.data.max_gt_segments))
        t.add_row("Devices",       str(cfg.lightning.devices))
        console.print()
        console.print(Panel(
            t,
            title="[bold white on dark_green]  GEODiffusion — Conditional Flow Matching  [/]",
            border_style="bright_green",
            box=box.DOUBLE_EDGE,
            padding=(1, 2),
        ))
        console.print()

    # Ensure output directories exist before loggers/checkpoints write files.
    _ensure_dirs(cfg)

    # ── resolve checkpoint path ───────────────────────────────────────────────
    # ckpt_path=None means fresh run; any non-None value means resume mode.
    ckpt_path = None

    # cfg.checkpoint.checkpoint_path accepts either:
    #   - "last" (resolve to runs/checkpoints/<run_id>/last.ckpt)
    #   - explicit checkpoint path
    raw = cfg.checkpoint.checkpoint_path if cfg.checkpoint is not None else None
    if raw:
        if str(raw).lower() == "last":
            candidate = Path(cfg.paths.checkpoints_folder) / cfg.experiment_run_id / "last.ckpt"
            ckpt_path = str(candidate)
        else:
            ckpt_path = str(raw)
        if is_rank0:
            console.print(f"[cyan]Resuming from:[/] {ckpt_path}")

    # TensorBoard version name is derived from checkpoint source so resume runs
    # are distinguishable (e.g., resume_last, resume_epoch_497_val_loss_0.0027).
    is_resume = ckpt_path is not None
    if is_resume:
        ckpt_stem = Path(ckpt_path).stem
        safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in ckpt_stem)
        version = f"resume_{safe_stem}"
    else:
        version = ""

    # ── loggers ───────────────────────────────────────────────────────────────
    # Custom logger hides auto "epoch" scalar and skips hparams table spam.
    loggers = [
        _CleanTBLogger(
            save_dir=cfg.lightning.logger.tensorboard_log_dir,
            name=cfg.experiment_run_id,
            version=version,
        ),
    ]

    # ── modules ───────────────────────────────────────────────────────────────
    # DataModule owns dataset construction and DataLoader configuration.
    data_module = VectorDiffusionDataModule(cfg)

    # LightningModule encapsulates model, losses, train/val steps, and eval visuals.
    lightning_module = VectorFlowLightningModule(cfg)

    # ── callbacks ─────────────────────────────────────────────────────────────
    # Checkpoint callbacks + one-line epoch summary.
    callbacks, best_ckpt = _setup_callbacks(cfg)
    # Skip _CompactModelSummary due to Lightning version compatibility issues
    callbacks += [_EpochSummaryCallback()]

    # ── trainer ───────────────────────────────────────────────────────────────
    # Core Trainer args are sourced from Hydra config for reproducible launches.
    trainer_kwargs: dict = {
        "max_epochs": cfg.training.num_epochs,
        "accelerator": cfg.lightning.accelerator,
        "devices": cfg.lightning.devices,
        "num_nodes": cfg.lightning.num_nodes,
        "precision": cfg.lightning.precision,
        "check_val_every_n_epoch": cfg.lightning.check_val_every_n_epoch,
        "log_every_n_steps": cfg.lightning.log_every_n_steps,
        "callbacks": callbacks,
        "logger": loggers,
        "default_root_dir": cfg.paths.logs_folder,
        "num_sanity_val_steps": 0,
        "enable_model_summary": False,   # handled by _CompactModelSummary callback
    }

    # Strategy selection:
    # - For DDP, use explicit DDPStrategy with extended timeout to avoid watchdog
    #   failures during long validation/comprehensive eval phases.
    # - Otherwise pass strategy through directly.
    if cfg.lightning.strategy:
        if cfg.lightning.strategy == "ddp":
            # Use explicit DDPStrategy with a long timeout so that the NCCL
            # watchdog doesn't fire during slow validation or comprehensive eval.
            trainer_kwargs["strategy"] = DDPStrategy(
                timeout=datetime.timedelta(hours=6)
            )
        else:
            trainer_kwargs["strategy"] = cfg.lightning.strategy

    # Optional gradient clipping configured from training.gradient_clipping.
    # Disabled by default when value is 0 or missing.
    gc = cfg.training.get("gradient_clipping", 0.0)
    if gc and float(gc) > 0:
        trainer_kwargs["gradient_clip_val"] = float(gc)
        trainer_kwargs["gradient_clip_algorithm"] = "norm"

    # Build Trainer and start fit; ckpt_path controls fresh-vs-resume behavior.
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(lightning_module, datamodule=data_module, ckpt_path=ckpt_path)

    # Final run summary printed only once on rank 0.
    if is_rank0:
        console.print("[bold green]Training complete![/]")
        # best_ckpt is tracked by callback monitor (typically Loss/val).
        if best_ckpt.best_model_path:
            console.print(f"Best checkpoint: {best_ckpt.best_model_path}")
            console.print(f"Best val loss:   {best_ckpt.best_model_score:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Hydra entry point
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        train(cfg)
    except Exception as e:
        import traceback
        console.print(f"\n[bold red]Training failed:[/] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
