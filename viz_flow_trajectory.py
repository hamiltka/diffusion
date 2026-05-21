"""
Flow matching trajectory visualizer for GEODiffusion.

Shows x0 (noisy anchor) -> x_t (intermediate steps) -> x1 (prediction) vs GT
on real satellite images. Good for a report figure.

Usage:
    conda activate trace_geo
    python viz_flow_trajectory.py \
        --checkpoint runs/checkpoints/demo_1k/last.ckpt \
        --config_dir runs/hydra/demo_1k/.hydra \
        --n_images 4 \
        --euler_steps 6 \
        --out flow_trajectory.png
"""
from __future__ import annotations

import argparse
import os
import sys
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))


def _draw_segs(ax, segs, color, lw=1.2, alpha=1.0, zorder=3):
    """Draw line segments on ax.  segs: [N, 4] in [-1,1] coords."""
    for s in segs:
        x0_, y0_, x1_, y1_ = float(s[0]), float(s[1]), float(s[2]), float(s[3])
        # convert from [-1,1] to [0,1] display
        ax.plot(
            [(x0_ + 1) / 2, (x1_ + 1) / 2],
            [1 - (y0_ + 1) / 2, 1 - (y1_ + 1) / 2],   # y-flip: image top = y=1 in screen
            color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=zorder,
        )


def _make_figure(samples, euler_steps, model, flow, anchors, device, anchor_noise_std):
    """Produce a (n_images × (euler_steps+3)) grid figure."""
    n_img = len(samples)
    n_cols = euler_steps + 3   # x0, t=0.17, t=0.33, ..., x1, GT
    col_labels = ["x₀ (noisy anchors)"]
    for i in range(1, euler_steps):
        col_labels.append(f"t={i/euler_steps:.2f}")
    col_labels += ["x₁ (predicted)", "GT"]

    fig, axes = plt.subplots(
        n_img, n_cols,
        figsize=(2.8 * n_cols, 2.8 * n_img),
        squeeze=False,
        gridspec_kw=dict(hspace=0.05, wspace=0.05),
    )

    for row, s in enumerate(samples):
        img_np   = s["img_np"]           # H×W×3  float [0,1]
        gt_segs  = s["gt_segs"]          # [M, 4] tensor
        gt_b     = s["gt_b_full"].unsqueeze(0).to(device)
        inv_b    = s["inv_b"].unsqueeze(0).to(device)
        img_t    = s["img_tensor"].float().unsqueeze(0).to(device) / 255.0

        # Build noisy anchor x0
        x0 = anchors.generate_from_gt(gt_b, inv_b, anchor_noise_std)  # [1,N,5]

        # ── euler integration, saving each step ──────────────────────────────
        steps_xt = [x0[0].cpu()]   # store [N,5] at each step
        xt = x0.clone()
        dt = 1.0 / euler_steps
        with torch.no_grad():
            for step_i in range(euler_steps):
                t_val    = step_i / euler_steps
                t_tensor = torch.full((1,), t_val, device=device)
                v_pred   = model(xt, t_tensor, image=img_t)
                xt       = xt + dt * v_pred
                xt[..., :4] = xt[..., :4].clamp(-1.0, 1.0)
                steps_xt.append(xt[0].cpu())

        # ── draw each column ─────────────────────────────────────────────────
        for col in range(n_cols):
            ax = axes[row][col]
            ax.imshow(img_np, extent=[0, 1, 0, 1], origin="upper", aspect="auto")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis("off")

            if col < n_cols - 1:
                # flow step columns
                segs_t = steps_xt[col]          # [N, 5]
                active_mask = segs_t[:, 4] > 0.0
                inactive    = segs_t[~active_mask, :4]
                active      = segs_t[active_mask, :4]
                _draw_segs(ax, inactive, color="#666666", lw=0.6, alpha=0.35)
                _draw_segs(ax, active,   color="#FF00FF", lw=1.4, alpha=0.9)
            else:
                # GT column
                _draw_segs(ax, gt_segs, color="#00FFFF", lw=1.6, alpha=1.0)

            # column header on first row
            if row == 0:
                ax.set_title(col_labels[col], fontsize=7, pad=3)

    # shared legend
    leg_items = [
        mpatches.Patch(color="#FF00FF", label="predicted (active)"),
        mpatches.Patch(color="#666666", label="predicted (inactive)"),
        mpatches.Patch(color="#00FFFF", label="ground truth"),
    ]
    fig.legend(handles=leg_items, loc="lower center", ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True,  help="Path to .ckpt file")
    parser.add_argument("--config_dir",   required=True,  help="Path to .hydra/ dir containing config.yaml + overrides.yaml")
    parser.add_argument("--n_images",     type=int, default=4)
    parser.add_argument("--euler_steps",  type=int, default=6)
    parser.add_argument("--active_threshold", type=float, default=0.0)
    parser.add_argument("--out",          default="flow_trajectory.png")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── load config via hydra compose ─────────────────────────────────────────
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()

    overrides_path = os.path.join(args.config_dir, "overrides.yaml")
    overrides = []
    if os.path.exists(overrides_path):
        import yaml
        with open(overrides_path) as f:
            overrides = yaml.safe_load(f) or []

    with initialize_config_dir(config_dir=os.path.abspath(args.config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    # ── load model from checkpoint ────────────────────────────────────────────
    from geodiffusion.lightning.lightning_module import VectorFlowLightningModule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module = VectorFlowLightningModule.load_from_checkpoint(
        args.checkpoint, cfg=cfg, map_location=device
    )
    module.eval().to(device)
    model   = module.model
    flow    = module.flow
    anchors = module.anchors
    anchor_noise_std = module.anchor_noise_std

    # ── load dataset ──────────────────────────────────────────────────────────
    from geodiffusion.dataloader.data_module import VectorRoadDataModule
    dm = VectorRoadDataModule(cfg)
    dm.setup("fit")
    dataset = dm.val_dataset

    # pick random samples
    indices = random.sample(range(len(dataset)), min(args.n_images, len(dataset)))
    samples = []
    for idx in indices:
        s = dataset[idx]
        inv = s["invalid_mask"]
        samples.append(dict(
            img_tensor  = s["image"],
            img_np      = s["image"].float().numpy().transpose(1, 2, 0) / 255.0,
            gt_segs     = s["road_data"][~inv],
            gt_b_full   = s["road_data"],
            inv_b       = inv,
        ))

    # ── build figure ──────────────────────────────────────────────────────────
    print(f"Building trajectory figure ({len(samples)} images × {args.euler_steps} steps)…")
    with torch.no_grad():
        fig = _make_figure(samples, args.euler_steps, model, flow, anchors, device, anchor_noise_std)

    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
