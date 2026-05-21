"""
NGA Demo: Degradation & Recovery visualization.

Loads the latest demo_1k checkpoint, picks 6 density-stratified val samples
(2 sparse / 2 medium / 2 dense), applies low-noise degradation, runs full
Euler integration to recover, and saves a presentation-quality figure.

Usage:
    conda activate trace_geo
    python nga_demo.py \
        [--checkpoint runs/checkpoints/demo_1k/last.ckpt] \
        [--output     runs/inference/nga_demo.png] \
        [--euler_steps 10] \
        [--noise_std   0.05] \
        [--dpi         150]
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))

from geodiffusion.lightning.lightning_module import VectorFlowLightningModule
from geodiffusion.dataloader.dataset import VectorRoadDataset
from geodiffusion.metrics.segment_metrics import segment_density_bucket


# ── colour palette ─────────────────────────────────────────────────────────────
GT_COLOR   = "#FF6A00"   # orange — ground truth
DEG_COLOR  = "#00BFFF"   # sky blue — degraded (noisy) input
REC_COLOR  = "#00EE44"   # bright green — recovered prediction
EP_COLOR   = "#FFD700"   # yellow — endpoints


def _to_px(v: float | np.ndarray, H: int) -> float | np.ndarray:
    return (v + 1.0) / 2.0 * H


def draw_segs(ax, segs_norm: np.ndarray, H: int, color: str,
              lw: float = 1.8, alpha: float = 0.9, ep_size: float = 12.0):
    for seg in segs_norm:
        x1p = _to_px(seg[0], H); y1p = _to_px(seg[1], H)
        x2p = _to_px(seg[2], H); y2p = _to_px(seg[3], H)
        ax.plot([x1p, x2p], [y1p, y2p], color=color, lw=lw,
                alpha=alpha, solid_capstyle="round")
        ax.scatter([x1p, x2p], [y1p, y2p],
                   color=EP_COLOR, s=ep_size, zorder=5, linewidths=0)


# ── dataset helpers ────────────────────────────────────────────────────────────

def select_samples(dataset, n_per_bucket: int = 2) -> list[dict]:
    target  = {"sparse": n_per_bucket, "medium": n_per_bucket, "dense": n_per_bucket}
    found: dict = {k: [] for k in target}
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    for idx in indices:
        if all(len(found[k]) >= n_per_bucket for k in target):
            break
        try:
            s = dataset[idx]
        except Exception:
            continue
        inv      = s["invalid_mask"]
        gt_valid = s["road_data"][~inv]
        bucket   = segment_density_bucket(gt_valid)
        if bucket not in found or len(found[bucket]) >= n_per_bucket:
            continue
        found[bucket].append(dict(
            img_np    = s["image"].float().numpy().transpose(1, 2, 0) / 255.0,
            img_tensor= s["image"],
            gt_segs   = gt_valid,
            gt_b_full = s["road_data"],
            inv_b     = inv,
            bucket    = bucket,
        ))
    result = []
    for b in ("sparse", "medium", "dense"):
        result.extend(found[b])
    return result


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="runs/checkpoints/demo_1k/last.ckpt")
    parser.add_argument("--output",      default="runs/inference/nga_demo.png")
    parser.add_argument("--euler_steps", type=int,   default=10)
    parser.add_argument("--noise_std",   type=float, default=0.05)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--dpi",         type=int,   default=150)
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--mode",        default="gt_seeded", choices=["gt_seeded", "grid"],
                        help="gt_seeded: degrade GT then recover; grid: use spoke-wheel grid anchors directly")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load checkpoint ──────────────────────────────────────────────────────
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg  = OmegaConf.create(ckpt["hyper_parameters"]["cfg"])
    OmegaConf.set_struct(cfg, False)
    cfg.training.euler_steps_eval = args.euler_steps

    module = VectorFlowLightningModule(cfg)
    missing, unexpected = module.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}...")
    module.eval().to(device)
    print(f"  Loaded — epoch {ckpt.get('epoch', '?')}")

    # ── load val dataset ─────────────────────────────────────────────────────
    print("Building val dataset...")
    d = cfg.data
    val_ds = VectorRoadDataset(
        data_root=d.data_root,
        split="val",
        max_gt_segments=d.max_gt_segments,
        densify=bool(d.get("densify", True)),
        max_segment_length=float(d.get("max_segment_length", 0.10)),
        image_size=int(d.get("image_size", 512)),
        gsd_m=float(d.get("gsd_m", 1.0)),
        max_samples=d.get("max_val_samples", None),
        augment=False,
    )
    print(f"  Val samples: {len(val_ds)}")

    samples = select_samples(val_ds, n_per_bucket=2)
    print(f"  Selected {len(samples)} samples: " +
          ", ".join(s["bucket"] for s in samples))

    # ── run inference ────────────────────────────────────────────────────────
    results = []
    with torch.no_grad():
        for s in samples:
            gt_b  = s["gt_b_full"].unsqueeze(0).to(device)
            inv_b = s["inv_b"].unsqueeze(0).to(device)
            img_t = s["img_tensor"].float().unsqueeze(0).to(device) / 255.0

            # Source anchors: gt_seeded (degrade GT) or grid (spoke-wheel)
            if args.mode == "gt_seeded":
                x0 = module.anchors.generate_from_gt(gt_b, inv_b, args.noise_std)
            else:
                x0 = module.anchors.generate(batch_size=1, device=device)

            # Recovery: full Euler integration
            x1pred = module.flow.euler_integrate(
                x0, module.model, img_t,
                steps=args.euler_steps, device=device
            )
            # Suppress padded slots (only meaningful in gt_seeded mode)
            if args.mode == "gt_seeded":
                x1pred[:, :, 4] = x1pred[:, :, 4].masked_fill(inv_b, -10.0)

            valid = (~s["inv_b"]).numpy()
            x0_active = x0[0, :, :4].cpu().numpy()  # show all anchors in grid mode
            x1_active = x1pred[0, :, :4].cpu().numpy()[
                (x1pred[0, :, 4] > module.active_threshold).cpu().numpy()
            ]

            results.append(dict(
                img_np  = s["img_np"],
                gt_segs = s["gt_segs"].numpy(),       # [K, 4] valid GT
                x0_segs = x0_active,                  # [K, 4] degraded
                x1_segs = x1_active,                  # [M, 4] recovered active
                bucket  = s["bucket"],
            ))

    # ── figure ───────────────────────────────────────────────────────────────
    n    = len(results)
    cols = 4   # satellite+GT | degraded | recovered | overlay
    col_titles = ["Ground Truth", "Degraded Input", "Recovered Output", "GT vs Recovered"]

    cell = 3.8  # inches per cell
    fig, axes = plt.subplots(n, cols,
                             figsize=(cols * cell, n * cell),
                             dpi=args.dpi,
                             gridspec_kw={"wspace": 0.02, "hspace": 0.04})
    if n == 1:
        axes = axes[None, :]

    fig.patch.set_facecolor("#111111")

    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=9, color="white", pad=4)

    for row, r in enumerate(results):
        img  = np.clip(r["img_np"], 0, 1)
        H    = img.shape[0]
        gt   = r["gt_segs"]
        deg  = r["x0_segs"]
        rec  = r["x1_segs"]
        bkt  = r["bucket"]

        for c in range(cols):
            ax = axes[row, c]
            ax.imshow(img, aspect="equal", extent=[0, H, H, 0])
            ax.set_xlim(0, H)
            ax.set_ylim(H, 0)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            ax.set_facecolor("black")

        # Col 0: GT
        draw_segs(axes[row, 0], gt,  H, GT_COLOR,  lw=2.0)

        # Col 1: Degraded (noisy)
        draw_segs(axes[row, 1], deg, H, DEG_COLOR, lw=1.6)

        # Col 2: Recovered
        draw_segs(axes[row, 2], rec, H, REC_COLOR, lw=1.8)

        # Col 3: Overlay GT + Recovered
        draw_segs(axes[row, 3], gt,  H, GT_COLOR,  lw=2.0, alpha=0.6)
        draw_segs(axes[row, 3], rec, H, REC_COLOR, lw=1.6, alpha=0.85, ep_size=0)

        # Row label
        axes[row, 0].text(0.01, 0.98, bkt, transform=axes[row, 0].transAxes,
                          color="white", fontsize=7, va="top", ha="left",
                          bbox=dict(facecolor="#00000088", edgecolor="none", pad=2))

    # Legend
    legend_ax = fig.add_axes([0.01, 0.01, 0.30, 0.025])
    legend_ax.axis("off")
    legend_ax.set_facecolor("#111111")
    patches = [
        mpatches.Patch(color=GT_COLOR,  label="Ground Truth"),
        mpatches.Patch(color=DEG_COLOR, label=f"Degraded (σ={args.noise_std})"),
        mpatches.Patch(color=REC_COLOR, label=f"Recovered ({args.euler_steps}-step Euler)"),
    ]
    legend_ax.legend(handles=patches, loc="center left", ncol=3,
                     fontsize=7, framealpha=0, labelcolor="white",
                     handlelength=1.5)

    fig.suptitle(
        f"GEODiffusion — Road Map Degradation & Recovery  "
        f"(checkpoint epoch {ckpt.get('epoch', '?')}, noise σ={args.noise_std})",
        fontsize=11, color="white", y=0.995
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.975, bottom=0.04,
                        wspace=0.02, hspace=0.04)
    fig.patch.set_facecolor("#111111")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
