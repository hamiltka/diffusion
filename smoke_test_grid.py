"""
Smoke test for VectorFlowLightningModule._make_sample_grid.

Loads real satellite images + GT road segments from the USGS dataset,
builds 6 samples (2 sparse / 2 medium / 2 dense) with synthetic predictions
(noisy GT, no model required), then saves the figure to smoke_test_grid.png.

Run:
    python3 smoke_test_grid.py
"""
import numpy as np
import torch

from geodiffusion.lightning.lightning_module import VectorFlowLightningModule
from geodiffusion.dataloader.dataset import VectorRoadDataset

# ── dataset ─────────────────────────────────────────────────────────────────
DATA_ROOT = "/shared/femiani_shared/data/usgs_crops_512_trace_2_NAIP"
N_ANCHORS = 200

print("Loading dataset …")
ds = VectorRoadDataset(
    data_root=DATA_ROOT,
    split="val",
    max_gt_segments=1500,
    densify=False,
    augment=False,
    image_size=512,
    max_samples=500,
)
print(f"  {len(ds)} samples available")

# ── scan for 2 samples per density tier ──────────────────────────────────────
rng = np.random.default_rng(0)
tiers: dict[str, list[int]] = {"sparse": [], "medium": [], "dense": []}
for idx in range(min(len(ds), 500)):
    s = ds[idx]
    inv = s["invalid_mask"]
    n_gt = int((~inv).sum().item())
    tier = "sparse" if n_gt < 20 else ("medium" if n_gt < 75 else "dense")
    if len(tiers[tier]) < 2:
        tiers[tier].append(idx)
    if all(len(v) >= 2 for v in tiers.values()):
        break

for tier, idxs in tiers.items():
    if len(idxs) < 2:
        raise RuntimeError(f"Not enough '{tier}' samples in first 500 val items (found {len(idxs)})")

print(f"  Locked indices: {tiers}")


# ── build sample dicts with real images/GT + synthetic predictions ─────────
def _build_sample(idx: int, tier: str) -> dict:
    s       = ds[idx]
    img_np  = s["image"].float().numpy().transpose(1, 2, 0) / 255.0
    inv     = s["invalid_mask"]
    gt_segs = s["road_data"][~inv]                            # [K, 4]
    K       = gt_segs.shape[0]
    N       = N_ANCHORS

    # x0: random spokes-style anchors (uniform random in [-1, 1])
    x0_b = torch.from_numpy(
        rng.uniform(-0.95, 0.95, (N, 5)).astype(np.float32)
    )

    # matched: first min(K, N) anchors assigned to GT (cycling if K < N)
    M = min(K, N)
    anc_matched = torch.arange(M)
    targets_b = torch.zeros(N, 5, dtype=torch.float32)
    for i in range(M):
        targets_b[i, :4] = gt_segs[i % K, :4]

    # x1pred: noisy GT for matched anchors (active), random for unmatched (inactive)
    x1pred_b = torch.from_numpy(
        rng.uniform(-0.95, 0.95, (N, 5)).astype(np.float32)
    )
    for i in range(M):
        noise = torch.from_numpy(rng.normal(0, 0.03, (4,)).astype(np.float32))
        x1pred_b[i, :4] = gt_segs[i % K, :4] + noise
        x1pred_b[i, 4]  = float(rng.uniform(0.3, 1.0))
    x1pred_b[M:, 4] = torch.from_numpy(
        rng.uniform(-1.0, -0.05, (N - M,)).astype(np.float32)
    )

    return dict(
        img_np      = np.clip(img_np, 0, 1),
        gt_segs     = gt_segs,
        x0_b        = x0_b,
        targets_b   = targets_b,
        x1pred_b    = x1pred_b,
        anc_matched = anc_matched,
        sample_idx  = idx,
        tier        = tier,
    )


samples = []
for tier in ("sparse", "medium", "dense"):
    for idx in tiers[tier]:
        samples.append(_build_sample(idx, tier))

print(f"  Built {len(samples)} samples")

# ── call _make_sample_grid ───────────────────────────────────────────────────
class _Stub:
    active_threshold = 0.0

stub = _Stub()
fig = VectorFlowLightningModule._make_sample_grid(stub, samples, title="Smoke Test – Val Grid (real data)")

if fig is None:
    raise RuntimeError("_make_sample_grid returned None")

out_path = "smoke_test_grid.png"
fig.savefig(out_path, bbox_inches="tight", dpi=110)
print(f"Saved: {out_path}  ({fig.get_figwidth():.0f}×{fig.get_figheight():.0f} in @110dpi)")

