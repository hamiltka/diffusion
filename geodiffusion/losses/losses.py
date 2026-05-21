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

import math
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


class EndpointClusteringLoss(nn.Module):
    """Hinge repulsion for inter-segment endpoints that are too close.

    Pushes apart nearby endpoints from *different* predicted segments.
    Masked so that it does NOT fire near GT junction nodes — at true
    junctions multiple predicted endpoints must coincide, and the
    repulsion would conflict with GTEndpointAttractionLoss there.

    Args:
        cluster_radius:   hinge distance threshold (normalised coords)
        active_threshold: min active score for a segment to be considered
        gt_exclusion_radius: skip pairs where either endpoint is within
                            this distance of any GT endpoint (set == attraction
                            radius so the two losses have non-overlapping zones)
    """

    def __init__(
        self,
        cluster_radius: float = 0.03,
        active_threshold: float = 0.0,
        gt_exclusion_radius: float = 0.05,
    ):
        super().__init__()
        self.cluster_radius = cluster_radius
        self.active_threshold = active_threshold
        self.gt_exclusion_radius = gt_exclusion_radius

    def forward(
        self,
        x1_pred: torch.Tensor,      # [B, N, 5]
        gt_segs: torch.Tensor | None = None,     # [B, K, 4]  optional
        invalid_mask: torch.Tensor | None = None, # [B, K] True=padding
    ) -> torch.Tensor:
        B = x1_pred.shape[0]
        total = None
        count = 0

        for b in range(B):
            coords = x1_pred[b, :, :4]   # [N, 4]
            scores = x1_pred[b, :, 4]    # [N]

            act = scores > self.active_threshold
            if act.sum() < 2:
                continue

            c = coords[act]              # [M, 4]
            M = c.shape[0]
            # Stack all endpoints: first M rows = p1, next M rows = p2
            endpoints = torch.cat([c[:, :2], c[:, 2:4]], dim=0)  # [2M, 2]

            # Intra-segment mask (same segment's own two endpoints)
            intra = torch.zeros(2 * M, 2 * M, dtype=torch.bool, device=c.device)
            idx = torch.arange(M, device=c.device)
            intra[idx, idx + M] = True
            intra[idx + M, idx] = True
            intra.fill_diagonal_(True)   # self-pairs

            D = torch.cdist(endpoints, endpoints)  # [2M, 2M]

            # GT-proximity exclusion: if GT info is provided, mask out any
            # endpoint that is within gt_exclusion_radius of a GT node.
            near_gt = torch.zeros(2 * M, dtype=torch.bool, device=c.device)
            if gt_segs is not None and invalid_mask is not None:
                valid_gt = gt_segs[b, ~invalid_mask[b]]   # [V, 4]
                if valid_gt.shape[0] > 0:
                    gt_eps = torch.cat(
                        [valid_gt[:, :2], valid_gt[:, 2:4]], dim=0
                    ).float()                              # [2V, 2]
                    D_gt = torch.cdist(endpoints.float(), gt_eps)  # [2M, 2V]
                    near_gt = D_gt.min(dim=1).values < self.gt_exclusion_radius

            # Exclude pair if either endpoint is near a GT node
            excl = near_gt.unsqueeze(1) | near_gt.unsqueeze(0)  # [2M, 2M]

            # Unique inter-segment pairs within cluster_radius, upper-triangle
            triu = torch.triu(
                torch.ones(2 * M, 2 * M, dtype=torch.bool, device=c.device),
                diagonal=1,
            )
            candidate = triu & ~intra & ~excl & (D < self.cluster_radius)
            if not candidate.any():
                continue

            pairs = candidate.nonzero(as_tuple=False)  # [P, 2]
            dists = D[pairs[:, 0], pairs[:, 1]]
            loss_term = (self.cluster_radius - dists).relu().mean()

            total = loss_term if total is None else total + loss_term
            count += 1

        if total is None:
            return x1_pred.sum() * 0.0
        return total / max(count, 1)


class GTEndpointAttractionLoss(nn.Module):
    """Pull each active predicted endpoint toward the nearest GT endpoint.

    For every active predicted segment, each of its two endpoints is pulled
    toward the closest valid GT endpoint.  Only fires when the predicted
    endpoint is already within ``attraction_radius`` of some GT endpoint —
    preventing far-away endpoints from being pulled across the image.

    Gradient flows only through predicted endpoint coordinates; GT endpoints
    are detached constants (no-grad).

    Args:
        attraction_radius: radius gate in normalised coords (default 0.05)
        active_threshold:  min active score to consider a segment
    """

    def __init__(self, attraction_radius: float = 0.05, active_threshold: float = 0.0):
        super().__init__()
        self.attraction_radius = attraction_radius
        self.active_threshold = active_threshold

    def forward(
        self,
        x1_pred: torch.Tensor,     # [B, N, 5]
        gt_segs: torch.Tensor,     # [B, K, 4]
        invalid_mask: torch.Tensor, # [B, K] True = padding
    ) -> torch.Tensor:
        B = x1_pred.shape[0]
        total = None
        count = 0

        for b in range(B):
            valid_gt = gt_segs[b, ~invalid_mask[b]]   # [V, 4]
            if valid_gt.shape[0] == 0:
                continue
            # GT endpoints are targets — stop gradients flowing into gt_segs
            gt_endpoints = torch.cat(
                [valid_gt[:, :2], valid_gt[:, 2:4]], dim=0
            ).float().detach()                         # [2V, 2]

            scores = x1_pred[b, :, 4]
            act_mask = scores > self.active_threshold
            if act_mask.sum() == 0:
                continue

            pred_coords = x1_pred[b, act_mask, :4]    # [M, 4]  has grad
            pred_eps = torch.cat(
                [pred_coords[:, :2], pred_coords[:, 2:4]], dim=0
            ).float()                                  # [2M, 2]

            D = torch.cdist(pred_eps, gt_endpoints)    # [2M, 2V]
            min_dists, _ = D.min(dim=1)                # [2M]

            close = min_dists < self.attraction_radius
            if not close.any():
                continue

            loss_term = min_dists[close].mean()        # mean over close endpoints only
            total = loss_term if total is None else total + loss_term
            count += 1

        if total is None:
            return x1_pred.sum() * 0.0
        return total / max(count, 1)  # average over contributing batch items



def _perp_dist_point_to_line(
    pts: torch.Tensor,    # [P, 2]  query points
    line_p: torch.Tensor, # [L, 2]  point on each line
    line_d: torch.Tensor, # [L, 2]  unit direction of each line
) -> torch.Tensor:        # [P, L]  perpendicular distances
    """Perpendicular distance from each point to each infinite line."""
    # diff[p, l] = pts[p] - line_p[l]   →  [P, L, 2]
    diff = pts.unsqueeze(1) - line_p.unsqueeze(0)
    # project along line direction
    proj = (diff * line_d.unsqueeze(0)).sum(-1)   # [P, L]
    # parallel component
    parallel = proj.unsqueeze(-1) * line_d.unsqueeze(0)  # [P, L, 2]
    perp_vec = diff - parallel                            # [P, L, 2]
    return perp_vec.norm(dim=-1)                          # [P, L]


class CollinearFragmentationLoss(nn.Module):
    """Penalise pairs of near-collinear active segments with a close shared endpoint.

    A pair (i, j) enters the loss when:
      1. Unique pair (i < j) — no double-counting.
      2. Both segments are active.
      3. The minimum endpoint gap delta_ij < gap_threshold.
      4. Directions are near-parallel/anti-parallel: |cos(angle)| > cos(theta_tol).
      5. **Same-line check**: each endpoint of i is within tau_perp of segment j's
         supporting line (and vice-versa).  This prevents pulling together parallel
         segments on *different* roads.

    Loss = mean(delta_ij) over qualifying pairs — a direct distance pull so the
    gradient always extends the closer fragment endpoints toward each other.

    Args:
        gap_threshold:    max normalised endpoint gap to enter loss (default 0.03)
        angle_tol_deg:    max angular deviation from collinear in degrees (default 20)
        tau_perp:         max perpendicular distance to the opposite segment's
                          supporting line for the same-line check (default 0.02)
        active_threshold: min active score for a segment to be included
    """

    def __init__(
        self,
        gap_threshold: float = 0.03,
        angle_tol_deg: float = 20.0,
        tau_perp: float = 0.02,
        active_threshold: float = 0.0,
    ):
        super().__init__()
        self.gap_threshold = gap_threshold
        self.cos_tol = math.cos(math.radians(angle_tol_deg))  # |cos| > this → collinear
        self.tau_perp = tau_perp
        self.active_threshold = active_threshold

    def forward(self, x1_pred: torch.Tensor) -> torch.Tensor:  # [B, N, 5]
        B = x1_pred.shape[0]
        total = None
        count = 0

        for b in range(B):
            scores = x1_pred[b, :, 4]
            mask = scores > self.active_threshold
            if mask.sum() < 2:
                continue

            coords = x1_pred[b, mask, :4]   # [M, 4]
            M = coords.shape[0]
            if M < 2:
                continue

            p1 = coords[:, :2]   # [M, 2]
            p2 = coords[:, 2:4]  # [M, 2]

            # ── Minimum endpoint gap across all 4 endpoint combos ────────────
            d11 = torch.cdist(p1, p1)   # [M, M]
            d12 = torch.cdist(p1, p2)
            d21 = torch.cdist(p2, p1)
            d22 = torch.cdist(p2, p2)
            min_gap = torch.stack([d11, d12, d21, d22], dim=0).min(dim=0).values  # [M, M]

            # ── Upper-triangle unique pairs within gap threshold ─────────────
            triu = torch.triu(
                torch.ones(M, M, dtype=torch.bool, device=coords.device), diagonal=1
            )
            near_mask = triu & (min_gap < self.gap_threshold)
            if not near_mask.any():
                continue

            # ── Direction unit vectors ───────────────────────────────────────
            dirs = (p2 - p1)                                    # [M, 2]
            dirs_n = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-6)  # [M, 2]

            # ── Collinearity: |cos(angle)| > cos(theta_tol) ─────────────────
            cos_ab = dirs_n @ dirs_n.t()                        # [M, M]
            collinear_mask = cos_ab.abs() > self.cos_tol

            candidate_mask = near_mask & collinear_mask
            if not candidate_mask.any():
                continue

            # ── Same-line check via perpendicular distance ───────────────────
            # For each candidate pair (i, j) check:
            #   endpoints of i are close to line through j, AND
            #   endpoints of j are close to line through i.
            # Use midpoints of each segment as the line anchor.
            mid = (p1 + p2) / 2.0                               # [M, 2]

            # perp[a, b] = perp dist from endpoints-of-a to line-of-b
            # endpoints of i: p1[i] and p2[i];  line of j: anchor=mid[j], dir=dirs_n[j]
            perp_p1_to_j = _perp_dist_point_to_line(p1, mid, dirs_n)  # [M, M]: [i, j]
            perp_p2_to_j = _perp_dist_point_to_line(p2, mid, dirs_n)  # [M, M]
            # at least one endpoint of i must be close to line of j
            perp_i_to_j = torch.minimum(perp_p1_to_j, perp_p2_to_j)  # [M, M]

            perp_p1_to_i = _perp_dist_point_to_line(p1, mid, dirs_n).t()  # [M, M]: [j, i] → transpose to [i, j]
            perp_p2_to_i = _perp_dist_point_to_line(p2, mid, dirs_n).t()
            perp_j_to_i = torch.minimum(perp_p1_to_i, perp_p2_to_i)  # [M, M]

            same_line = (perp_i_to_j < self.tau_perp) & (perp_j_to_i < self.tau_perp)

            final_mask = candidate_mask & same_line
            if not final_mask.any():
                continue

            dists = min_gap[final_mask]
            loss_term = dists.mean()
            total = loss_term if total is None else total + loss_term
            count += 1

        if total is None:
            return x1_pred.sum() * 0.0
        return total / max(count, 1)


def build_loss_fns(cfg_loss) -> dict:
    """Instantiate all loss functions from config."""
    return {
        "segment":      DistanceLoss(),
        "active":       ActiveLoss(),
        "connectivity": ConnectivityLoss(node_eps=float(cfg_loss.get("node_eps", 0.02))),
        "endpoint_clustering": EndpointClusteringLoss(
            cluster_radius=float(cfg_loss.get("endpoint_cluster_radius", 0.03)),
            active_threshold=float(cfg_loss.get("endpoint_cluster_active_threshold", 0.0)),
            gt_exclusion_radius=float(cfg_loss.get("gt_endpoint_attraction_radius", 0.05)),
        ),
        "gt_endpoint_attraction": GTEndpointAttractionLoss(
            attraction_radius=float(cfg_loss.get("gt_endpoint_attraction_radius", 0.05)),
            active_threshold=float(cfg_loss.get("endpoint_cluster_active_threshold", 0.0)),
        ),
        "collinear_fragmentation": CollinearFragmentationLoss(
            gap_threshold=float(cfg_loss.get("collinear_gap_threshold", 0.03)),
            angle_tol_deg=float(cfg_loss.get("collinear_angle_tol_deg", 20.0)),
            tau_perp=float(cfg_loss.get("collinear_tau_perp", 0.02)),
            active_threshold=float(cfg_loss.get("endpoint_cluster_active_threshold", 0.0)),
        ),
    }

