#!/usr/bin/env python
"""
Analyze segment count and length distributions across a dataset split.

Usage:
    python analyze_segments.py [--data-root PATH] [--split val|train|test] [--out-dir PATH]

Outputs (written to <out_dir>/):
    per_sample.csv          — per-sample segment count, total + per highway tag
    highway_summary.csv     — aggregate length stats per highway type
    count_distribution.png  — histogram of valid segments per sample
    length_distribution.png — normalized segment length histograms (overall + per highway type)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PROJ_NETWORK", "OFF")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "NO")

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

# Make project importable when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geodiffusion.dataloader.dataset import VALID_HIGHWAY_TYPES
from geodiffusion.preprocessing.densify import densify_segments

NO_TAG_LABEL = "(no_tag)"  # highway property absent — kept permissively by dataset


def _parse_segments(
    road_path: str,
    ds,
    img_size: int,
    *,
    apply_artifact_filter: bool = True,
    densify: bool = False,
    max_seg_len: float = 0.06,
) -> list[tuple[str, float]]:
    """Return (highway_label, normalized_length) for every valid segment in one GeoJSON file.

    Mirrors the filtering logic in VectorRoadDataset._parse_roads:
      - features with no 'highway' property  → kept, labelled NO_TAG_LABEL
      - features whose 'highway' is in VALID_HIGHWAY_TYPES → kept
      - anything else                         → skipped

    apply_artifact_filter: drop segments with length² > 2.0 (spans > √2 in [-1,1] space)
      or length² < 1e-8 (zero-length).  Matches the filter added to dataset._parse_roads.
    densify: split segments longer than max_seg_len into equal sub-segments.
    """
    if not os.path.exists(road_path):
        return []
    with open(road_path) as f:
        data = json.load(f)

    half = img_size / 2.0
    crs4326 = CRS.from_epsg(4326)
    results: list[tuple[str, float]] = []

    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        hw_raw = props.get("highway", "")
        if hw_raw and hw_raw not in VALID_HIGHWAY_TYPES:
            continue  # invalid tag — dataset skips these
        hw_label = hw_raw if hw_raw else NO_TAG_LABEL

        geom = feature.get("geometry")
        if geom is None:
            continue

        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        if gtype == "LineString":
            lines = [coords]
        elif gtype == "MultiLineString":
            lines = coords
        elif gtype == "Polygon":
            lines = coords  # exterior + interior rings
        elif gtype == "MultiPolygon":
            lines = [ring for poly in coords for ring in poly]
        else:
            continue

        for line in lines:
            if len(line) < 2:
                continue
            lons = [c[0] for c in line]
            lats = [c[1] for c in line]
            try:
                xs, ys = warp_transform(crs4326, ds.crs, lons, lats)
            except Exception:
                continue
            pts = [ds.index(x, y) for x, y in zip(xs, ys)]
            raw: list[tuple[float, float, float, float]] = []
            for i in range(len(pts) - 1):
                # pixel coords: pts[i] = (row, col)
                x1, y1 = float(pts[i][1]), float(pts[i][0])
                x2, y2 = float(pts[i + 1][1]), float(pts[i + 1][0])
                # normalize to [-1, 1]  (same space as max_segment_length)
                nx1 = max(-1.0, min(1.0, (x1 - half) / half))
                ny1 = max(-1.0, min(1.0, (y1 - half) / half))
                nx2 = max(-1.0, min(1.0, (x2 - half) / half))
                ny2 = max(-1.0, min(1.0, (y2 - half) / half))
                raw.append((nx1, ny1, nx2, ny2))

            # Apply the same artifact + zero-length filter as dataset._parse_roads
            if apply_artifact_filter:
                raw = [
                    s for s in raw
                    if 1e-8 < (s[2] - s[0]) ** 2 + (s[3] - s[1]) ** 2 <= 2.0
                ]

            # Optionally densify
            if densify:
                raw = densify_segments(raw, max_length=max_seg_len)

            for seg in raw:
                length = math.sqrt((seg[2] - seg[0]) ** 2 + (seg[3] - seg[1]) ** 2)
                results.append((hw_label, length))

    return results


def analyze(
    data_root: str,
    split: str,
    out_dir: Path,
    *,
    densify: bool = False,
    max_seg_len: float = 0.06,
) -> None:
    base = Path(data_root) / split
    sat_paths = sorted(base.glob("*_sat.jpg"))
    densify_note = f" (densify=True, max_seg_len={max_seg_len})" if densify else ""
    print(f"Found {len(sat_paths)} samples in {base}{densify_note}")

    # per-sample segment counts by highway type
    per_sample: dict[str, dict[str, int]] = {}
    # all lengths keyed by highway type
    lengths_by_hw: dict[str, list[float]] = defaultdict(list)

    try:
        from tqdm import tqdm
        iterator = tqdm(sat_paths, desc="Parsing", unit="sample")
    except ImportError:
        iterator = sat_paths

    for p in iterator:
        sample_id = p.stem.removesuffix("_sat")
        road_path = str(p.with_name(sample_id + "_osm.geojson"))
        try:
            with rasterio.open(str(p)) as ds:
                segs = _parse_segments(
                    road_path, ds, ds.width,
                    densify=densify, max_seg_len=max_seg_len,
                )
        except Exception as e:
            print(f"  [skip] {sample_id}: {e}")
            continue

        counts: dict[str, int] = defaultdict(int)
        for hw, length in segs:
            counts[hw] += 1
            lengths_by_hw[hw].append(length)
        per_sample[sample_id] = dict(counts)

    if not per_sample:
        print("No samples processed — check data_root and split.")
        return None

    all_lengths = []
    for lengths in lengths_by_hw.values():
        all_lengths.extend(lengths)

    if not all_lengths:
        print(f"  Split '{split}' has no road annotations — skipping plots and summary.")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    all_hw_types = sorted(per_sample[next(iter(per_sample))].keys() |
                          {hw for c in per_sample.values() for hw in c})

    # ── per_sample.csv ────────────────────────────────────────────────────────
    totals = []
    with open(out_dir / "per_sample.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "total_segments"] + all_hw_types)
        for sid, counts in sorted(per_sample.items()):
            total = sum(counts.values())
            totals.append(total)
            writer.writerow([sid, total] + [counts.get(hw, 0) for hw in all_hw_types])

    # ── highway_summary.csv ───────────────────────────────────────────────────
    all_lengths = []
    for lengths in lengths_by_hw.values():
        all_lengths.extend(lengths)

    with open(out_dir / "highway_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "highway_type", "total_segments",
            "mean_length", "median_length",
            "p25_length", "p75_length", "p95_length", "max_length",
        ])
        # overall row first
        arr = np.array(all_lengths)
        writer.writerow(["ALL", len(arr),
                         f"{arr.mean():.5f}", f"{np.median(arr):.5f}",
                         f"{np.percentile(arr, 25):.5f}", f"{np.percentile(arr, 75):.5f}",
                         f"{np.percentile(arr, 95):.5f}", f"{arr.max():.5f}"])
        for hw in sorted(lengths_by_hw):
            arr = np.array(lengths_by_hw[hw])
            writer.writerow([hw, len(arr),
                             f"{arr.mean():.5f}", f"{np.median(arr):.5f}",
                             f"{np.percentile(arr, 25):.5f}", f"{np.percentile(arr, 75):.5f}",
                             f"{np.percentile(arr, 95):.5f}", f"{arr.max():.5f}"])

    # ── plots ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    totals_arr = np.array(totals)

    # 1. Segment count distribution per sample
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(totals_arr, bins=60, edgecolor="black", linewidth=0.3, color="steelblue")
    ax.axvline(totals_arr.mean(),   color="red",    linestyle="--", label=f"mean={totals_arr.mean():.0f}")
    ax.axvline(np.median(totals_arr), color="orange", linestyle="--", label=f"median={np.median(totals_arr):.0f}")
    ax.set_xlabel("Valid segments per sample (before max_gt_segments trim)")
    ax.set_ylabel("# samples")
    ax.set_title(f"Segment count distribution — {split}  ({len(totals_arr):,} samples)\n"
                 f"min={totals_arr.min()}  max={totals_arr.max()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "count_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Segment length distributions — overall + one panel per highway type
    hw_sorted = sorted(lengths_by_hw)
    n_panels = 1 + len(hw_sorted)           # overall + one per tag
    ncols = 3
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    def _length_panel(ax, lengths, title, default_max=0.1):
        arr = np.array(lengths)
        ax.hist(arr, bins=80, edgecolor="none", color="steelblue", alpha=0.85)
        ax.axvline(default_max, color="red", linestyle="--", linewidth=1.2,
                   label=f"default max_len={default_max}")
        p95 = np.percentile(arr, 95)
        ax.axvline(p95, color="green", linestyle=":", linewidth=1.2,
                   label=f"p95={p95:.3f}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Normalized length [-1,1] space", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    _length_panel(axes[0], all_lengths, f"ALL tags (n={len(all_lengths):,})")
    for i, hw in enumerate(hw_sorted, start=1):
        _length_panel(axes[i], lengths_by_hw[hw], f"{hw}  (n={len(lengths_by_hw[hw]):,})")
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Segment length distributions — {split}", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "length_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── console summary ───────────────────────────────────────────────────────
    arr_all = np.array(all_lengths)
    print(f"\n{'─'*60}")
    print(f"Split: {split}   Samples: {len(totals_arr):,}")
    print(f"Segments/sample  min={totals_arr.min()}  max={totals_arr.max()}  "
          f"mean={totals_arr.mean():.1f}  median={np.median(totals_arr):.1f}  "
          f"p95={np.percentile(totals_arr,95):.0f}  p99={np.percentile(totals_arr,99):.0f}")
    print(f"Total segments  : {len(all_lengths):,}")
    print(f"Length (norm)   : mean={arr_all.mean():.4f}  median={np.median(arr_all):.4f}  "
          f"p95={np.percentile(arr_all,95):.4f}  max={arr_all.max():.4f}")
    print(f"{'─'*60}")
    print(f"Outputs → {out_dir}/")
    print(f"  per_sample.csv, highway_summary.csv")
    print(f"  count_distribution.png, length_distribution.png")

    # return stats for combined summary
    return {
        "split": split,
        "samples": len(totals_arr),
        "seg_min": int(totals_arr.min()),
        "seg_max": int(totals_arr.max()),
        "seg_mean": float(totals_arr.mean()),
        "seg_median": float(np.median(totals_arr)),
        "seg_p95": float(np.percentile(totals_arr, 95)),
        "seg_p99": float(np.percentile(totals_arr, 99)),
        "total_segments": len(all_lengths),
        "len_mean": float(arr_all.mean()),
        "len_median": float(np.median(arr_all)),
        "len_p95": float(np.percentile(arr_all, 95)),
        "len_max": float(arr_all.max()),
    }


def write_combined_summary(all_stats: list[dict], root_out: Path) -> None:
    """Write a single CSV + comparison PNG across all processed splits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # CSV
    fields = list(all_stats[0].keys())
    with open(root_out / "combined_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_stats)

    # Comparison bar chart: key percentiles of segment counts per split
    splits = [s["split"] for s in all_stats]
    x = np.arange(len(splits))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- left: segment count percentiles ---
    for i, (label, key) in enumerate([("median", "seg_median"), ("p95", "seg_p95"),
                                       ("p99", "seg_p99"), ("max", "seg_max")]):
        vals = [s[key] for s in all_stats]
        ax1.bar(x + i * width, vals, width, label=label)
    ax1.set_xticks(x + 1.5 * width)
    ax1.set_xticklabels(splits)
    ax1.set_ylabel("Segments per sample")
    ax1.set_title("Segment count percentiles by split")
    ax1.legend()

    # --- right: segment length percentiles ---
    for i, (label, key) in enumerate([("median", "len_median"), ("p95", "len_p95"),
                                       ("max", "len_max")]):
        vals = [s[key] for s in all_stats]
        ax2.bar(x + i * width, vals, width, label=label)
    ax2.axhline(0.1, color="red", linestyle="--", linewidth=1, label="default max_len=0.1")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(splits)
    ax2.set_ylabel("Normalized length")
    ax2.set_title("Segment length percentiles by split")
    ax2.legend()

    fig.suptitle("Combined distribution summary across splits", fontsize=13)
    fig.tight_layout()
    fig.savefig(root_out / "combined_summary.png", dpi=150)
    plt.close(fig)
    print(f"\nCombined summary → {root_out}/combined_summary.csv  +  combined_summary.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/home/hamiltka/usgs-osm-crops-512-1m-trace-2.0",
                    help="Root directory of the dataset (contains train/, val/, test/)")
    ap.add_argument("--split", default="all",
                    choices=["train", "val", "test", "all"],
                    help="Split to analyze, or 'all' to run train+val+test")
    ap.add_argument("--out-dir", default=None,
                    help="Parent output directory (default: data_distribution_test/)")
    ap.add_argument("--densify", action="store_true",
                    help="Apply densification (same as dataset densify=True)")
    ap.add_argument("--max-segment-length", type=float, default=0.06,
                    help="Max segment length for densification (default 0.06)")
    args = ap.parse_args()

    root_out = Path(args.out_dir) if args.out_dir else Path(__file__).parent
    if args.densify:
        root_out = root_out / "densified"
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    all_stats = []
    for split in splits:
        split_dir = root_out / split
        stats = analyze(
            args.data_root, split, split_dir,
            densify=args.densify,
            max_seg_len=args.max_segment_length,
        )
        if stats:
            all_stats.append(stats)

    if len(all_stats) > 1:
        write_combined_summary(all_stats, root_out)


if __name__ == "__main__":
    main()
