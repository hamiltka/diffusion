# GEODiffusion Research Findings

Running log of code changes, config tuning, and experimental results.
Each entry states **what** changed, **why** (evidence), and **which files** were touched.

---

## 2026-05-21 ~09:00 — Session 1: Dataset Recovery & Refactor

### Context
`geodiffusion/dataloader/dataset.py` was accidentally deleted. It was reconstructed
from `__pycache__/dataset.cpython-310.pyc` bytecode and refactored during recovery.

### Changes

| # | What | Why | Files |
|---|------|-----|-------|
| 1 | Rewrote `dataset.py` using Path ops, assert validation, sentinel padding `(2.0, 2.0, 2.0, 2.0)` | File lost; reconstruction improved readability and made pad sentinel explicit | `geodiffusion/dataloader/dataset.py` |
| 2 | Added `_apply_exclusion_csv_filter()` with `crop_id`/`sample_id` column auto-detection | Prior code had no resilience to column name variations | `geodiffusion/dataloader/dataset.py` |
| 3 | `VALID_HIGHWAY_TYPES` defined as a module-level frozenset | Centralised filtering logic shared with analysis scripts | `geodiffusion/dataloader/dataset.py` |

---

## 2026-05-21 ~11:00 — Session 2: Distribution Analysis

### Context
Built a standalone analysis script to understand the ground-truth segment distribution
before attempting to fix training. Running the overfit experiment (`overfit_gt_seeded`)
was producing poor metrics: `mean_active_count≈984` vs `mean_gt_count≈82`,
`match_fraction=0.054`, `Active/precision=0.06`.

### Distribution results (measured)

| Split | Samples | Seg median | Seg p95 | Seg p99 | Seg max | Len median |
|-------|---------|------------|---------|---------|---------|------------|
| train | 46,410  | 59         | 189     | 292     | 882     | 0.065      |
| val   | 2,053   | 66         | 211     | 332     | 779     | 0.063      |
| test  | 14,025  | —          | —       | —       | —       | (unlabeled)|

Post-densification (`densify=True`, `max_seg_len=0.06`) val results:

| Metric | Value |
|--------|-------|
| Median segments/image | 194 |
| p95 | 456 |
| p99 | **710** |
| Max | 1928 |
| Mean length | 0.0479 |
| Median length | 0.0522 |
| Max length | 0.0600 (exactly the cap — correct) |

4.4% of val samples (91/2053) have 0 road segments.

### Changes

| # | What | Why | Files |
|---|------|-----|-------|
| 4 | Created `data_distribution_test/analyze_segments.py` | Need to measure segment count + length distributions to calibrate model hyperparameters | `data_distribution_test/analyze_segments.py` |
| 5 | Added `--densify` / `--max-segment-length` flags to analysis script; apply same artifact filter as dataset | Required to measure *post-densification* segment counts to set `max_gt_segments` correctly | `data_distribution_test/analyze_segments.py` |

---

## 2026-05-21 ~13:00 — Session 3: Training Fixes

### Root cause analysis

TensorBoard after 44 epochs of `overfit_gt_seeded`:
- `Val/active_loss = 1.53` (dominates total loss of ~1.34)
- `Active/precision = 0.06` — model activates ~984/1536 spokes but only ~83 are GT
- `Active/recall = 0.73` — "predict everything active" collapse

Five independent causes identified and fixed:

---

### Fix 1 — `generate_from_gt` random spoke → deterministic closest-tip

**File:** `geodiffusion/anchors/spoke_wheel.py`

**Problem:** `generate_from_gt` flagged a **random** spoke (of 24 per cell) as `active=+1`
in `x0`. But `build_targets` (Hungarian matching) assigns spokes by minimum endpoint
distance — almost always a *different* spoke. Result: with 24 spokes/cell, the
probability of agreement was ≈1/24 = 4%.

When they disagree:
- `v_gt_active[spoke_A]` = −1 − (+1) = **−2** (deactivate the wrongly-flagged spoke)
- `v_gt_active[spoke_B]` = +1 − (−1) = **+2** (activate the Hungarian-matched spoke)

The model received contradictory ±2 active-channel gradients every step instead of the
intended ≈0 (gt_seeded should already have correct active flags, requiring no change).

**Fix:** Replace `random_k = torch.randint(0, K, ...)` with deterministic selection of
the spoke in the nearest cell whose **tip** is closest (PI distance) to either GT
endpoint. This aligns `generate_from_gt`'s spoke assignment with what Hungarian will
pick, so `v_gt_active ≈ 0` throughout training and gradients flow cleanly to the
coordinate channels.

```python
# Before (random — misaligned with Hungarian ≈96% of the time):
random_k = torch.randint(0, K, nearest_cell.shape, device=device)

# After (closest tip — aligns with Hungarian):
near_tips = spoke_tips[nearest_cell]          # [B, N_gt, K, 2]
d_p1 = (near_tips - gt_p1).pow(2).sum(-1)    # [B, N_gt, K]
d_p2 = (near_tips - gt_p2).pow(2).sum(-1)
best_k = torch.minimum(d_p1, d_p2).argmin(-1) # [B, N_gt]
```

---

### Fix 2 — `active_pos_weight` 13 → 18

**File:** `configs/loss/default.yaml`

**Problem:** True imbalance = (1536 − 82.6) / 82.6 ≈ **17.6:1** (measured match
fraction 5.4%). Previous value of 13 was calibrated for an assumed 7% match rate.
Formula: `(1 − match_fraction) / match_fraction`.

**Fix:** `active_pos_weight: 13.0 → 18.0`

Note: with Fix 1 in place, active velocities are ≈0, so pos_weight primarily governs
the small residual gradient. The correction is still correct in principle.

---

### Fix 3 — Filter artifact segments (length > √2)

**File:** `geodiffusion/dataloader/dataset.py` (`_parse_roads`)

**Problem:** Distribution analysis found segments with normalised length up to 2.498
in [-1,1] space. √2 ≈ 1.414 is the maximum physically meaningful length (half-diagonal
of the [-1,1]² image). Longer segments are reprojection artefacts: the road extends
beyond the image boundary, is clipped to [-1,1], and the resulting segment spans edge
to edge. Its "endpoints" are image-boundary clips, not real road junctions.
Without densification these artefacts would produce ≥23 garbage sub-segments each.

**Fix:** Added filter after normalisation, before densification:
```python
norm = [
    seg for seg in norm
    if (seg[2]-seg[0])**2 + (seg[3]-seg[1])**2 > 1e-8   # drop zero-length
    and (seg[2]-seg[0])**2 + (seg[3]-seg[1])**2 <= 2.0   # drop > √2
]
```

---

### Fix 4 — Drop 0-segment samples

**File:** `geodiffusion/dataloader/dataset.py` (`__getitem__`)

**Problem:** 4.4% of val samples (91/2053) have zero road annotations after filtering.
These contribute zero gradient signal but waste batch capacity.

**Fix:** Return `index=-1` for samples where `not segments`. The existing collator
already drops `index=-1` entries via `_collate()`.

---

### Fix 5 — Drop instead of truncate for over-limit samples

**File:** `geodiffusion/dataloader/dataset.py` (`_pad_or_trim`, `__getitem__`)

**Problem:** `_pad_or_trim` randomly shuffled and truncated samples with more segments
than `max_gt_segments`. Truncated GT is **worse** than no GT: the model sees a partial
road network and has no way to distinguish "no road here" from "road exists but was
removed from the batch". This corrupts the matching loss for affected samples.

**Fix:** Removed the shuffle+truncate. `_pad_or_trim` now asserts `len <= n`.
`__getitem__` returns `index=-1` (dropped by collator) for samples exceeding the limit.
Users must set `max_gt_segments` to cover their data's p99 post-densification count.

---

### Fix 6 — `max_gt_segments` 1500 → 800

**Files:** `configs/data/usgs_crops_512_trace_2_demo1k.yaml`,
`configs/data/usgs_crops_512_trace_2.yaml`,
`configs/data/usgs_crops_512_trace_2_final500.yaml`

**Problem:** Post-densification p99 = 710 (measured on val). `max_gt_segments=1500`
was 2.1× too large: 75% of tensor slots were padding, and the Hungarian cost matrix
was 1536 × 1500 with ~1100 empty GT rows. This wastes memory and compute.

**Fix:** `max_gt_segments: 1500 → 800` (covers p99=710 with 13% buffer).
`~1%` of samples above 800 segments are now dropped rather than truncated.

---

### Fix 7 — Enable densification in demo1k config

**File:** `configs/data/usgs_crops_512_trace_2_demo1k.yaml`

**Problem:** `densify: false` — long segments (max raw length 2.498) were passing
through as-is. Without densification, a 0.5-length segment matched to one spoke
required the spoke to travel 4× its own length. Also: consistency — all other
data configs have `densify: true`.

**Fix:** `densify: false → true`

---

### Fix 8 — `spoke_length` 0.12 → 0.06

**File:** `configs/anchors/spoke_wheel.yaml`

**Problem:** `spoke_length=0.12` while `max_segment_length=0.06` after densification
means spokes are **2.5× longer** than the mean GT segment (mean=0.0479, ratio=2.5×).
Every spoke needed a systematic "shrink" velocity component, consuming model capacity
before any meaningful direction or position learning could occur.
Scale ratio: `0.12 / 0.0479 = 2.5×` (old) vs `0.06 / 0.0479 = 1.25×` (new).

**Fix:** `spoke_length: 0.12 → 0.06`

This also matches spoke scale to the densification target, making p2 velocities
primarily about direction and position rather than length correction.

---

## Pending / Future Work

- **Highway type weighting in loss**: residential dominates (56.8%), motorways are
  rarest but longest. Requires adding a highway-class integer channel to `road_data`
  and weight lookup in `DistanceLoss`/`ActiveLoss`. Defer until active-channel precision
  recovers from the spoke-selection fix.
- **Re-run overfit experiment** with all above fixes and observe TensorBoard.
  Key metrics to watch: `Train/active_accuracy`, `Val/active_loss`, `mean_active_count`.
- **Run train-split post-densification analysis** to confirm p99 is consistent with val.
