"""
GEODiffusion evaluation script.

Computes segment-level Precision / Recall / F1 by comparing predicted
GeoJSON files against the dataset GT for a given split.

Usage:
    python eval.py \
        --pred_dir  runs/inference/epoch157_test \
        --data_root /shared/femiani_shared/data/usgs_crops_512_trace_2_NAIP \
        [--split    test] \
        [--thresholds 0.05 0.10] \
        [--image_size 512] \
        [--output   runs/inference/epoch157_test/eval_results.csv] \
        [--workers  8]

Output:
    Prints a summary table to stdout.
    Writes a per-image CSV to --output (if given).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from geodiffusion.dataloader.dataset import VectorRoadDataset
from geodiffusion.metrics.segment_metrics import segment_precision_recall_f1, apls, topo


# ── helpers ────────────────────────────────────────────────────────────────────

def load_pred_geojson(path: str, image_size: int = 512) -> torch.Tensor:
    """Load a *_pred.geojson and return segments in normalized [-1, 1] coords.

    infer.py writes pixel-space coordinates (0 to image_size).
    Normalization: norm = pixel / (image_size / 2) - 1
    """
    with open(path) as f:
        data = json.load(f)

    half = image_size / 2.0
    segs = []
    for feat in data.get("features", []):
        coords = feat["geometry"]["coordinates"]
        if len(coords) < 2:
            continue
        x1, y1 = coords[0][0], coords[0][1]
        x2, y2 = coords[1][0], coords[1][1]
        # pixel → [-1, 1]
        x1n = max(-1.0, min(1.0, x1 / half - 1.0))
        y1n = max(-1.0, min(1.0, y1 / half - 1.0))
        x2n = max(-1.0, min(1.0, x2 / half - 1.0))
        y2n = max(-1.0, min(1.0, y2 / half - 1.0))
        if x1n == x2n and y1n == y2n:
            continue  # degenerate
        segs.append([x1n, y1n, x2n, y2n])

    if segs:
        return torch.tensor(segs, dtype=torch.float32)
    return torch.zeros((0, 4), dtype=torch.float32)


def eval_one(
    sample_id: str,
    gt_segs: torch.Tensor,       # [N, 4] in [-1,1]
    pred_path: str | None,
    thresholds: list[float],
    image_size: int,
) -> dict:
    """Evaluate a single image. Returns a dict of metrics."""
    result: dict = {"sample_id": sample_id}

    if pred_path is None or not os.path.exists(pred_path):
        # No prediction → zero precision, zero recall
        for tau in thresholds:
            key = f"t{int(tau * 100):03d}"
            result[f"prec_{key}"] = 0.0
            result[f"rec_{key}"]  = 0.0
            result[f"f1_{key}"]   = 0.0
        result["pred_count"] = 0
        result["gt_count"]   = int(gt_segs.shape[0])
        result["apls"]       = float("nan")
        result["topo_prec"]  = float("nan")
        result["topo_rec"]   = float("nan")
        result["topo_f1"]    = float("nan")
        return result

    pred_segs = load_pred_geojson(pred_path, image_size=image_size)
    result["pred_count"] = int(pred_segs.shape[0])
    result["gt_count"]   = int(gt_segs.shape[0])

    for tau in thresholds:
        p, r, f = segment_precision_recall_f1(pred_segs, gt_segs, threshold=tau)
        key = f"t{int(tau * 100):03d}"
        result[f"prec_{key}"] = float(p)
        result[f"rec_{key}"]  = float(r)
        result[f"f1_{key}"]   = float(f)

    result["apls"]      = apls(pred_segs, gt_segs)
    tp, tr, tf          = topo(pred_segs, gt_segs)
    result["topo_prec"] = tp
    result["topo_rec"]  = tr
    result["topo_f1"]   = tf

    return result


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEODiffusion evaluation")
    parser.add_argument("--pred_dir",   required=True,
                        help="Directory containing *_pred.geojson prediction files")
    parser.add_argument("--data_root",  required=True,
                        help="Dataset root (contains train/ val/ test/ splits)")
    parser.add_argument("--split",      default="test",
                        help="Dataset split to evaluate [test]")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.10],
                        help="Distance thresholds in [-1,1] coords [0.05 0.10]")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size in pixels [512]")
    parser.add_argument("--output",     default=None,
                        help="Path to write per-image CSV results")
    parser.add_argument("--workers",    type=int, default=4,
                        help="Parallel workers for loading GT [4]")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Evaluate on at most this many images (for quick checks)")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    thresholds = sorted(args.thresholds)

    print(f"Loading GT from : {args.data_root} / {args.split}")
    print(f"Predictions from: {pred_dir}")
    print(f"Thresholds      : {thresholds}")

    # Load dataset (no augment, densify=True to match training normalization)
    ds = VectorRoadDataset(
        data_root=args.data_root,
        split=args.split,
        densify=True,
        max_segment_length=0.06,
        augment=False,
        image_size=args.image_size,
        use_exclusion_csv=False,
    )

    n_total = len(ds)
    if args.max_samples is not None:
        n_total = min(n_total, args.max_samples)
    print(f"Evaluating {n_total} images ...")

    results: list[dict] = []

    for idx in range(n_total):
        sample = ds[idx]
        sample_id = ds.sample_ids[idx]

        # GT segments — drop padded/invalid rows
        gt_raw  = sample["road_data"]      # [N, 4]
        inv_mask = sample["invalid_mask"]  # [N] bool
        gt_segs = gt_raw[~inv_mask]        # [M, 4]

        pred_path_str = str(pred_dir / f"{sample_id}_pred.geojson")

        row = eval_one(
            sample_id=sample_id,
            gt_segs=gt_segs,
            pred_path=pred_path_str,
            thresholds=thresholds,
            image_size=args.image_size,
        )
        results.append(row)

        if (idx + 1) % 500 == 0 or (idx + 1) == n_total:
            key = f"f1_t{int(thresholds[0] * 100):03d}"
            mean_f1   = np.nanmean([r[key]      for r in results])
            mean_apls = np.nanmean([r["apls"]   for r in results])
            mean_topo = np.nanmean([r["topo_f1"] for r in results])
            print(f"  [{idx+1}/{n_total}]  F1@{thresholds[0]:.2f}: {mean_f1:.4f}  APLS: {mean_apls:.4f}  TOPO-F1: {mean_topo:.4f}")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"{'Threshold':>12}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    print("-" * 68)
    for tau in thresholds:
        key = f"t{int(tau * 100):03d}"
        mean_p = np.nanmean([r[f"prec_{key}"] for r in results])
        mean_r = np.nanmean([r[f"rec_{key}"]  for r in results])
        mean_f = np.nanmean([r[f"f1_{key}"]   for r in results])
        print(f"  τ = {tau:.3f}   {mean_p:>10.4f}  {mean_r:>10.4f}  {mean_f:>10.4f}")

    mean_apls   = np.nanmean([r["apls"]      for r in results])
    mean_topo_p = np.nanmean([r["topo_prec"] for r in results])
    mean_topo_r = np.nanmean([r["topo_rec"]  for r in results])
    mean_topo_f = np.nanmean([r["topo_f1"]   for r in results])
    print("-" * 68)
    print(f"  APLS                                          {mean_apls:>10.4f}")
    print(f"  TOPO             {mean_topo_p:>10.4f}  {mean_topo_r:>10.4f}  {mean_topo_f:>10.4f}")

    n_missing = sum(1 for r in results if r["pred_count"] == 0 and r["gt_count"] > 0)
    print("=" * 68)
    print(f"Images evaluated : {len(results)}")
    print(f"Missing preds    : {n_missing}")
    print(f"Mean pred segs   : {np.mean([r['pred_count'] for r in results]):.1f}")
    print(f"Mean GT segs     : {np.mean([r['gt_count']   for r in results]):.1f}")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["sample_id", "pred_count", "gt_count"]
        for tau in thresholds:
            key = f"t{int(tau * 100):03d}"
            fieldnames += [f"prec_{key}", f"rec_{key}", f"f1_{key}"]
        fieldnames += ["apls", "topo_prec", "topo_rec", "topo_f1"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nPer-image results → {out_path}")


if __name__ == "__main__":
    main()
