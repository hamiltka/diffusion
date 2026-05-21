"""
Bipartite anchor ↔ GT-segment matching.

For each image in a batch, we must assign every GT segment to exactly one
unique anchor.  The cost metric is the *permutation-invariant endpoint MSE*
which treats the two endpoints as an unordered pair:

    cost(anchor_a, gt_b) =
        min(
            ||a.p1 − b.p1||² + ||a.p2 − b.p2||²,
            ||a.p1 − b.p2||² + ||a.p2 − b.p1||²
        ) / 2

This prevents the model from having to learn a canonical endpoint order.

The assignment is solved with the Hungarian algorithm
(scipy.optimize.linear_sum_assignment on the rectangular cost matrix).
"""
from __future__ import annotations

import torch
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


def _pairwise_pi_cost(
    anchors_xy: torch.Tensor,   # [N, 4]  x1,y1,x2,y2 of anchors
    gt_xy: torch.Tensor,        # [M, 4]  x1,y1,x2,y2 of GT segments
) -> torch.Tensor:
    """Return [M, N] permutation-invariant endpoint cost matrix."""
    # Forward assignment: a.p1↔b.p1, a.p2↔b.p2
    # cost_fwd[m, n] = ||a_n.p1 - b_m.p1||² + ||a_n.p2 - b_m.p2||²
    ap1 = anchors_xy[:, :2]   # [N, 2]
    ap2 = anchors_xy[:, 2:4]
    gp1 = gt_xy[:, :2]        # [M, 2]
    gp2 = gt_xy[:, 2:4]

    # [M, N] squared distances
    d11 = torch.cdist(gp1, ap1) ** 2  # [M, N]
    d22 = torch.cdist(gp2, ap2) ** 2
    d12 = torch.cdist(gp1, ap2) ** 2
    d21 = torch.cdist(gp2, ap1) ** 2

    cost_fwd = (d11 + d22) / 2.0
    cost_rev = (d12 + d21) / 2.0
    return torch.minimum(cost_fwd, cost_rev)   # [M, N]


def build_targets(
    anchors:      torch.Tensor,         # [B, N, 5]  (x1,y1,x2,y2,active)
    gt_segs:      torch.Tensor,         # [B, max_gt, 4]
    invalid_mask: torch.Tensor,         # [B, max_gt]  True = padded
    node_eps:     float = 0.02,         # proximity threshold for junction detection
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """
    Match anchors to GT segments, build per-anchor targets, and detect junctions.

    Returns:
        targets:        [B, N, 5]  — target positions for flow matching
        matched_mask:   [B, N]     — True where anchor was matched to a GT segment
        junction_pairs: [P_total, 5]  int64 — one row per junction pair:
            [batch_idx, anchor_i, endpoint_i, anchor_j, endpoint_j]
            anchor_* in [0, N), endpoint_* in {0=p1, 1=p2}.
            Empty tensor of shape [0, 5] when no junctions exist in the batch.
            Two endpoints are paired when their GT counterparts are within
            node_eps of each other (i.e. they share a road junction node).
    """
    B, N, _ = anchors.shape
    device = anchors.device

    targets      = anchors.clone()       # start from anchor positions
    targets[:, :, 4] = -1.0             # default active = -1 (inactive)

    all_matched      = []  # accumulate [b, anchor] rows across batch
    all_junction_pairs = []  # accumulate [b, ep_i, ep_j] rows across batch

    if not _SCIPY_OK:
        raise RuntimeError(
            "scipy is required for anchor matching.  "
            "Install with: pip install scipy"
        )

    for b in range(B):
        valid_idx = (~invalid_mask[b]).nonzero(as_tuple=True)[0]  # [M]
        M = len(valid_idx)
        if M == 0:
            continue

        gt_xy  = gt_segs[b][valid_idx]              # [M, 4]
        anc_xy = anchors[b, :, :4]                  # [N, 4]

        # Cost matrix [M, N] — match each GT to one unique anchor
        cost    = _pairwise_pi_cost(anc_xy, gt_xy)  # [M, N]
        cost_np = cost.detach().cpu().numpy()

        gt_ind, anchor_ind = linear_sum_assignment(cost_np)  # each of length min(M,N)

        anc_t  = torch.tensor(anchor_ind, device=device, dtype=torch.long)  # [M_b]
        gt_t   = valid_idx[gt_ind]                                           # [M_b] original GT indices
        b_col  = torch.full((len(anc_t), 1), b, device=device, dtype=torch.long)
        all_matched.append(torch.cat([b_col, anc_t.unsqueeze(1)], dim=1))   # [M_b, 2]

        # Write matched GT endpoints and mark active
        targets[b, anc_t, :4] = gt_segs[b, gt_t]
        targets[b, anc_t, 4]  = 1.0

        # ── Junction detection ─────────────────────────────────────────────
        # Which pairs of matched GT segments share a road node?  We check
        # endpoint proximity once here so ConnectivityLoss never needs to
        # redo this search.  gt_xy[gt_ind] gives matched GT endpoints in
        # the same order as anchor_ind (the assignment order).
        M_b = len(gt_ind)
        if M_b >= 2:
            anc_idx   = anc_t                                                       # [M_b] already a tensor
            gt_matched = gt_xy[gt_ind]                                              # [M_b, 4]
            gt_eps     = gt_matched.view(M_b, 2, 2).permute(1, 0, 2).reshape(2 * M_b, 2)  # [2M_b, 2]

            D      = torch.cdist(gt_eps.float(), gt_eps.float())                    # [2M_b, 2M_b]
            eye    = torch.eye(2 * M_b, device=device, dtype=torch.bool)
            shared = (D < node_eps) & ~eye                                          # [2M_b, 2M_b]

            if shared.any():
                # Each local index k encodes: anchor = anc_idx[k % M_b], endpoint = k // M_b (0=p1, 1=p2)
                p     = shared.nonzero(as_tuple=False)          # [P, 2]  local indices in [0, 2*M_b)
                b_col = torch.full((p.shape[0], 1), b, device=device, dtype=torch.long)
                anc   = anc_idx[p % M_b]                        # [P, 2]  anchor index in [0, N)
                ep    = (p // M_b).long()                        # [P, 2]  0=p1, 1=p2
                # interleave as [batch, anc_i, ep_i, anc_j, ep_j]
                row = torch.cat([b_col, anc[:, :1], ep[:, :1], anc[:, 1:], ep[:, 1:]], dim=1)  # [P, 5]
                all_junction_pairs.append(row)

        # Unmatched anchors: collapse to spoke centre, active = -1
        all_anc = torch.arange(N, device=device)
        matched_set = anc_t if len(anc_t) > 0 else torch.zeros(0, dtype=torch.long, device=device)
        unmatched_mask = torch.ones(N, dtype=torch.bool, device=device)
        unmatched_mask[matched_set] = False
        unmatched = unmatched_mask.nonzero(as_tuple=True)[0]
        cx = (anchors[b, unmatched, 0] + anchors[b, unmatched, 2]) / 2.0
        cy = (anchors[b, unmatched, 1] + anchors[b, unmatched, 3]) / 2.0
        targets[b, unmatched, 0] = cx
        targets[b, unmatched, 1] = cy
        targets[b, unmatched, 2] = cx
        targets[b, unmatched, 3] = cy
        # active = -1 already set

    # Matched indices: [M_total, 2] — one row per matched anchor.
    if all_matched:
        matched_indices = torch.cat(all_matched, dim=0)   # [M_total, 2]
    else:
        matched_indices = torch.zeros(0, 2, dtype=torch.long, device=device)

    # Junction pairs: [P_total, 5] — one row per junction pair.
    # Empty tensor (shape [0, 5]) when no junctions exist anywhere in the batch.
    if all_junction_pairs:
        junction_pairs = torch.cat(all_junction_pairs, dim=0)   # [P_total, 5]
    else:
        junction_pairs = torch.zeros(0, 5, dtype=torch.long, device=device)

    return targets, matched_indices, junction_pairs
