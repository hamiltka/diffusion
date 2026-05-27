"""
Loss functions for vector road-network flow matching.

Three independent components, each configurable via cfg.loss weights:

(A) DistanceLoss
    Permutation-invariant MSE on predicted endpoint velocities.
    Applied only to matched anchors (those assigned to a GT segment).

(B) ActiveLoss
    MSE regression: the active channel (index 4) should flow toward
    +1 for matched anchors and -1 for unmatched.
    Applied to all anchors.

(C) ConnectivityLoss
    Penalises predicted endpoint positions that should converge to the
    same road node but diverge.  A "shared node" is any pair of GT segment
    endpoints within `node_eps` of each other.  The predicted final position
    is estimated as x1_pred = anchor + v_pred (Euler step from source).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DistanceLoss(nn.Module):
    """Permutation-invariant endpoint MSE on velocity predictions.

    For each matched anchor we evaluate:

        min( ||v_pred_p1 - v_gt_p1||² + ||v_pred_p2 - v_gt_p2||²,
             ||v_pred_p1 - v_gt_p2||² + ||v_pred_p2 - v_gt_p1||² ) / 2

    The minimum over both orderings removes the need for a canonical
    endpoint direction.
    """

    def forward(
        self,
        v_pred:          torch.Tensor,  # [B, N, 4]  predicted velocity (endpoints only)
        v_gt:            torch.Tensor,  # [B, N, 4]  GT velocity (endpoints only)
        matched_indices: torch.Tensor,  # [M_total, 2]  rows: [batch_idx, anchor_idx]
    ) -> torch.Tensor:
        if matched_indices.shape[0] == 0:
            return v_pred.sum() * 0.0

        b  = matched_indices[:, 0]
        ai = matched_indices[:, 1]
        vp = v_pred[b, ai]   # [M_total, 4]
        vg = v_gt[b, ai]

        vp_p1, vp_p2 = vp[:, :2], vp[:, 2:]   # predicted velocities for each endpoint
        vg_p1, vg_p2 = vg[:, :2], vg[:, 2:]   # GT velocities for each endpoint

        # fwd and rev are intentionally kept as [M] vectors, not scalars.
        # torch.minimum below picks the better endpoint ordering *per anchor*
        # independently — some anchors need forward, others need reversed.
        # Collapsing to scalars first would force one global ordering for all M anchors.
        # Forward ordering: p1→p1, p2→p2
        fwd = (vp_p1 - vg_p1).pow(2).sum(-1) + (vp_p2 - vg_p2).pow(2).sum(-1)
        # Reversed ordering: p1→p2, p2→p1  (GT segments have no canonical direction)
        rev = (vp_p1 - vg_p2).pow(2).sum(-1) + (vp_p2 - vg_p1).pow(2).sum(-1)

        return (torch.minimum(fwd, rev) / 2.0).mean()


class ActiveLoss(nn.Module):
    """Weighted MSE regression of the active-channel velocity toward ±1 targets.

    Applied to every anchor (not just matched).

    Class imbalance: ~86% of anchors are inactive (target=-1), only ~14% are
    active (target=+1).  Plain MSE makes the model collapse to all-inactive.
    ``pos_weight`` upweights the matched (active=+1) anchors so their gradient
    contribution matches that of the inactive majority.

    Args:
        pos_weight: multiplier on matched-anchor loss terms.  Set to
            (1 - match_fraction) / match_fraction ≈ 6.0 for 14% match rate.
    """

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        v_pred_active: torch.Tensor,  # [B, N]  predicted velocity for active channel
        v_gt_active: torch.Tensor,    # [B, N]  GT velocity for active channel
    ) -> torch.Tensor:
        sq_err = (v_pred_active - v_gt_active) ** 2
        if self.pos_weight != 1.0:
            # matched anchors (v_gt_active > 0) get pos_weight, unmatched get 1.0
            weight = 1.0 + (self.pos_weight - 1.0) * (v_gt_active > 0).float()
            return (sq_err * weight).mean()
        return sq_err.mean()


class ConnectivityLoss(nn.Module):
    """Penalise endpoint divergence for GT shared nodes.

    Two predicted endpoint positions are "supposed to converge" when their
    corresponding GT endpoints are within `node_eps` of each other.
    We compute the pairwise distance between all pairs of such predicted
    endpoints and penalise it.

    To keep this tractable we process each sample in the batch independently
    and skip samples with no shared nodes.
    """

    def __init__(self, node_eps: float = 0.02):
        super().__init__()
        self.node_eps = node_eps

    def forward(
        self,
        x1_pred:        torch.Tensor,  # [B, N, 5]  predicted final positions (x0 + v_pred)
        junction_pairs: torch.Tensor,  # [P_total, 5]  rows: [batch_idx, anc_i, ep_i, anc_j, ep_j]
    ) -> torch.Tensor:
        """
        Penalise endpoint divergence at known junction nodes.

        junction_pairs is a [P_total, 5] tensor produced by build_targets:
          col 0: batch index
          col 1: anchor index of first endpoint
          col 2: 0 (p1) or 1 (p2) for first endpoint
          col 3: anchor index of second endpoint
          col 4: 0 (p1) or 1 (p2) for second endpoint
        Shape [0, 5] means no junctions in this batch.
        """
        if junction_pairs.shape[0] == 0:
            return x1_pred.sum() * 0.0   # grad-safe zero; no junctions in this batch

        # [B, N, 2, 2]: natural shape — anchor axis, then p1/p2, then xy
        coords = x1_pred[:, :, :4].clamp(-1.0, 1.0).view(
            x1_pred.shape[0], x1_pred.shape[1], 2, 2
        )

        b   = junction_pairs[:, 0]
        ai  = junction_pairs[:, 1]
        epi = junction_pairs[:, 2]   # 0=p1, 1=p2
        aj  = junction_pairs[:, 3]
        epj = junction_pairs[:, 4]

        diff = coords[b, ai, epi] - coords[b, aj, epj]   # [P_total, 2]
        return (diff ** 2).sum(-1).mean()

