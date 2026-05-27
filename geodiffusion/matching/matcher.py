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


def _pairwise_pi_cost(anchors_xy: torch.Tensor, gt_xy: torch.Tensor) -> torch.Tensor:
    """
    Compute [M, N] permutation-invariant endpoint cost matrix between anchors and GT segments.
    Args:
        anchors_xy: [N, 4]  x1,y1,x2,y2 of anchors
        gt_xy: [M, 4]  x1,y1,x2,y2 of GT segments
    Returns:
        [M, N] cost matrix
    """
    ap1, ap2 = anchors_xy[:, :2], anchors_xy[:, 2:4]
    gp1, gp2 = gt_xy[:, :2], gt_xy[:, 2:4]
    # Compute both endpoint assignments and take the minimum
    cost_fwd = (torch.cdist(gp1, ap1) ** 2 + torch.cdist(gp2, ap2) ** 2) / 2.0
    cost_rev = (torch.cdist(gp1, ap2) ** 2 + torch.cdist(gp2, ap1) ** 2) / 2.0
    return torch.minimum(cost_fwd, cost_rev)


def build_targets(
    anchors: torch.Tensor,         # [B, N, 5]  (x1,y1,x2,y2,active)
    gt_segs: torch.Tensor,         # [B, max_gt, 4]
    invalid_mask: torch.Tensor,    # [B, max_gt]  True = padded
    node_eps: float = 0.02,        # proximity threshold for junction detection
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Match anchors to GT segments, build per-anchor targets, and detect junctions.
    Args:
        anchors: [B, N, 5] (x1,y1,x2,y2,active)
        gt_segs: [B, max_gt, 4]
        invalid_mask: [B, max_gt] (True = padded)
        node_eps: proximity threshold for junction detection
    Returns:
        targets: [B, N, 5] — target positions for flow matching
        matched_indices: [M_total, 2] — [batch_idx, anchor_idx]
        junction_pairs: [P_total, 5] — [batch, anc_i, ep_i, anc_j, ep_j]
    """
    B, N, _ = anchors.shape
    device = anchors.device
    targets = anchors.clone()
    targets[:, :, 4] = -1.0  # default inactive
    all_matched, all_junction_pairs = [], []
    if not _SCIPY_OK:
        raise RuntimeError("scipy is required for anchor matching. Install with: pip install scipy")
    for b in range(B):
        valid_idx = (~invalid_mask[b]).nonzero(as_tuple=True)[0]
        if len(valid_idx) == 0: continue
        gt_xy, anc_xy = gt_segs[b][valid_idx], anchors[b, :, :4]
        cost_np = _pairwise_pi_cost(anc_xy, gt_xy).detach().cpu().numpy()
        gt_ind, anchor_ind = linear_sum_assignment(cost_np)
        anc_t = torch.tensor(anchor_ind, device=device, dtype=torch.long)
        gt_t = valid_idx[gt_ind]
        b_col = torch.full((len(anc_t), 1), b, device=device, dtype=torch.long)
        all_matched.append(torch.cat([b_col, anc_t.unsqueeze(1)], dim=1))
        targets[b, anc_t, :4] = gt_segs[b, gt_t]
        targets[b, anc_t, 4] = 1.0
        # Junction detection: find endpoints within node_eps
        M_b = len(gt_ind)
        if M_b >= 2:
            anc_idx = anc_t
            gt_matched = gt_xy[gt_ind]
            gt_eps = gt_matched.view(M_b, 2, 2).permute(1, 0, 2).reshape(2 * M_b, 2)
            D = torch.cdist(gt_eps.float(), gt_eps.float())
            shared = (D < node_eps) & ~torch.eye(2 * M_b, device=device, dtype=torch.bool)
            if shared.any():
                p = shared.nonzero(as_tuple=False)
                b_col = torch.full((p.shape[0], 1), b, device=device, dtype=torch.long)
                anc = anc_idx[p % M_b]
                ep = (p // M_b).long()
                row = torch.cat([b_col, anc[:, :1], ep[:, :1], anc[:, 1:], ep[:, 1:]], dim=1)
                all_junction_pairs.append(row)
        # Unmatched anchors: collapse to spoke centre
        unmatched_mask = torch.ones(N, dtype=torch.bool, device=device)
        unmatched_mask[anc_t] = False
        unmatched = unmatched_mask.nonzero(as_tuple=True)[0]
        cx = (anchors[b, unmatched, 0] + anchors[b, unmatched, 2]) / 2.0
        cy = (anchors[b, unmatched, 1] + anchors[b, unmatched, 3]) / 2.0
        targets[b, unmatched, 0:4:2] = cx.unsqueeze(1)
        targets[b, unmatched, 1:4:2] = cy.unsqueeze(1)
    matched_indices = torch.cat(all_matched, dim=0) if all_matched else torch.zeros(0, 2, dtype=torch.long, device=device)
    junction_pairs = torch.cat(all_junction_pairs, dim=0) if all_junction_pairs else torch.zeros(0, 5, dtype=torch.long, device=device)
    return targets, matched_indices, junction_pairs
