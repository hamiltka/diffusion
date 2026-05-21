"""
Visualize predicted vs GT road segments for a random sample of test images.

Produces a grid: [Satellite] [GT (valid types)] [Prediction] per row.

Usage:
    python viz_predictions.py \
        --pred_dir   runs/inference/epoch157_test \
        --data_root  /shared/femiani_shared/data/usgs_crops_512_trace_2_NAIP \
        [--split     test] \
        [--n         20] \
        [--seed      42] \
        [--output    runs/inference/epoch157_test/viz_grid.png] \
        [--eval_csv  runs/inference/epoch157_test/eval_results.csv]

If --eval_csv is given, F1@0.05 is shown under each prediction panel.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from geodiffusion.dataloader.dataset import VectorRoadDataset


# ── helpers ────────────────────────────────────────────────────────────────────

def load_pred_geojson(path: str, image_size: int = 512) -> np.ndarray:
    """Load *_pred.geojson → [M,4] in pixel coords."""
    if not os.path.exists(path):
        return np.zeros((0, 4))
    with open(path) as f:
        data = json.load(f)
    segs = []
    for feat in data.get("features", []):
        coords = feat["geometry"]["coordinates"]
        if len(coords) < 2:
            continue
        segs.append([coords[0][0], coords[0][1], coords[1][0], coords[1][1]])
    return np.array(segs, dtype=np.float32) if segs else np.zeros((0, 4))


def norm_to_px(segs_norm: np.ndarray, image_size: int = 512) -> np.ndarray:
    """Convert [-1,1] normalised coords to pixel coords."""
    if segs_norm.shape[0] == 0:
        return segs_norm
    half = image_size / 2.0
    px = segs_norm.copy()
    px[:, [0, 2]] = (segs_norm[:, [0, 2]] + 1.0) * half
    px[:, [1, 3]] = (segs_norm[:, [1, 3]] + 1.0) * half
    return px


def draw_segments(ax, segs_px: np.ndarray, color: str, lw: float = 1.2, alpha: float = 0.9):
    for x1, y1, x2, y2 in segs_px:
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw,
                alpha=alpha, solid_capstyle="round")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize GEODiffusion predictions")
    parser.add_argument("--pred_dir",   required=True)
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--split",      default="test")
    parser.add_argument("--n",          type=int,   default=20,
                        help="Number of images to show [20]")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--output",     default=None,
                        help="Output PNG path [pred_dir/viz_grid.png]")
    parser.add_argument("--image_size", type=int,   default=512)
    parser.add_argument("--eval_csv",   default=None,
                        help="eval_results.csv to annotate F1 scores")
    parser.add_argument("--sort_by",    default=None,
                        choices=["f1_asc", "f1_desc", "random"],
                        help="How to pick images: random (default), f1_asc (worst first), f1_desc (best first)")
    args = parser.parse_args()

    pred_dir   = Path(args.pred_dir)
    output     = Path(args.output) if args.output else pred_dir / "viz_grid.png"
    sort_by    = args.sort_by or "random"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ── Load dataset for GT + sat images ──────────────────────────────────────
    print(f"Loading dataset from {args.data_root} / {args.split} ...")
    ds = VectorRoadDataset(
        data_root=args.data_root,
        split=args.split,
        densify=True,
        max_segment_length=0.06,
        augment=False,
        image_size=args.image_size,
        use_exclusion_csv=False,
    )
    print(f"  {len(ds)} images in split")

    # ── Optionally load eval CSV for F1 annotations ───────────────────────────
    f1_map: dict[str, float] = {}
    if args.eval_csv and os.path.exists(args.eval_csv):
        with open(args.eval_csv) as f:
            for row in csv.DictReader(f):
                try:
                    f1_map[row["sample_id"]] = float(row["f1_t005"])
                except (KeyError, ValueError):
                    pass
        print(f"  Loaded F1 scores for {len(f1_map)} images from {args.eval_csv}")

    # ── Select images ──────────────────────────────────────────────────────────
    n = min(args.n, len(ds))
    rng = random.Random(args.seed)

    if sort_by == "random":
        indices = rng.sample(range(len(ds)), n)
    else:
        # Sort by F1 from eval CSV
        scored = [(i, f1_map.get(ds.sample_ids[i], 0.5)) for i in range(len(ds))]
        scored.sort(key=lambda x: x[1], reverse=(sort_by == "f1_desc"))
        indices = [i for i, _ in scored[:n]]
        # Shuffle within chosen set for display variety
        rng.shuffle(indices)

    print(f"Plotting {n} images (sort_by={sort_by}) ...")

    # ── Build figure ───────────────────────────────────────────────────────────
    ncols = 3   # sat | gt | pred
    nrows = n
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.5),
                             dpi=100)
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, ds_idx in enumerate(indices):
        sample    = ds[ds_idx]
        sample_id = ds.sample_ids[ds_idx]

        # Satellite image
        img_np = sample["image"].permute(1, 2, 0).numpy()   # [H,W,3] uint8

        # GT segments → pixel coords
        gt_raw  = sample["road_data"]
        inv     = sample["invalid_mask"]
        gt_norm = gt_raw[~inv].numpy()
        gt_px   = norm_to_px(gt_norm, args.image_size)

        # Predicted segments (already in pixel coords from infer.py)
        pred_path = pred_dir / f"{sample_id}_pred.geojson"
        pred_px   = load_pred_geojson(str(pred_path), args.image_size)

        ax_sat  = axes[row_idx, 0]
        ax_gt   = axes[row_idx, 1]
        ax_pred = axes[row_idx, 2]

        # ── Satellite panel ──────────────────────────────────────────────────
        ax_sat.imshow(img_np)
        ax_sat.set_title(f"{sample_id}", fontsize=7, pad=3)
        ax_sat.axis("off")

        # ── GT panel ────────────────────────────────────────────────────────
        ax_gt.imshow(img_np)
        draw_segments(ax_gt, gt_px, color="#00FFFF", lw=1.5)
        ax_gt.set_title(f"GT  ({len(gt_px)} segs)", fontsize=8, pad=3)
        ax_gt.axis("off")

        # ── Prediction panel ─────────────────────────────────────────────────
        ax_pred.imshow(img_np)
        draw_segments(ax_pred, pred_px, color="#FF00FF", lw=1.5)
        f1_str = ""
        if sample_id in f1_map:
            f1_str = f"  F1@τ.05={f1_map[sample_id]:.3f}"
        ax_pred.set_title(f"Pred ({len(pred_px)} segs){f1_str}", fontsize=8, pad=3)
        ax_pred.axis("off")

    # Legend
    gt_patch   = mpatches.Patch(color="#00FFFF", label="GT (valid road types)")
    pred_patch = mpatches.Patch(color="#FF00FF", label="Prediction")
    fig.legend(handles=[gt_patch, pred_patch], loc="lower center",
               ncol=2, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    plt.suptitle(
        f"GEODiffusion — epoch 157 predictions on {args.split} set\n"
        f"τ=0.05 ≈ 12.8px  |  cyan=GT  |  magenta=Pred",
        fontsize=11, y=1.002,
    )
    plt.tight_layout(pad=0.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output), bbox_inches="tight", dpi=100)
    print(f"\nSaved → {output}")


if __name__ == "__main__":
    main()
