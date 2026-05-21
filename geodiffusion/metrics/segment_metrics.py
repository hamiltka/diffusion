"""
Publication-quality evaluation metrics for vector road segment prediction.

All coordinates are in normalised [-1, 1] image space.
Distance threshold reference (512 px image):
    τ = 0.05  →  ~12.8 px   ≈ one lane width  (strict)
    τ = 0.10  →  ~25.6 px   ≈ two lane widths (lenient)

Core per-pair distance: permutation-invariant RMS endpoint error
    d(a, b) = sqrt( min(fwd_mse, rev_mse) )
    fwd_mse = (||p1_a − p1_b||² + ||p2_a − p2_b||²) / 2
    rev_mse = (||p1_a − p2_b||² + ||p2_a − p1_b||²) / 2

This equals the RMS of both endpoint errors under the best endpoint pairing,
allowing the model to predict segment directions without canonical ordering.
"""
from __future__ import annotations

import torch


def pi_dist_matrix(
    segs_a: torch.Tensor,   # [M, 4]  x1,y1,x2,y2
    segs_b: torch.Tensor,   # [N, 4]  x1,y1,x2,y2
) -> torch.Tensor:
    """[M, N] permutation-invariant RMS endpoint distance matrix.

    Entry [m, n] = the RMS endpoint error between segs_a[m] and segs_b[n]
    under the best endpoint pairing.

    Values are in the same units as the input coordinates (normalised [-1,1]).
    """
    ap1, ap2 = segs_a[:, :2], segs_a[:, 2:4]   # [M, 2]
    bp1, bp2 = segs_b[:, :2], segs_b[:, 2:4]   # [N, 2]

    d11 = torch.cdist(ap1, bp1) ** 2  # [M, N]
    d22 = torch.cdist(ap2, bp2) ** 2
    d12 = torch.cdist(ap1, bp2) ** 2
    d21 = torch.cdist(ap2, bp1) ** 2

    fwd = (d11 + d22) / 2.0
    rev = (d12 + d21) / 2.0
    return torch.minimum(fwd, rev).sqrt()   # [M, N]  in distance units


def chamfer_endpoints(
    pred_segs: torch.Tensor,   # [M, 4]  active predicted segments
    gt_segs:   torch.Tensor,   # [N, 4]  valid GT segments
) -> torch.Tensor:
    """Symmetric endpoint Chamfer distance.

    Decomposes each segment into 2 endpoints (4 points total per segment),
    then computes the symmetric nearest-neighbour Euclidean distance:

        Chamfer = 0.5 * (mean_pred_to_gt + mean_gt_to_pred)

    Returns scalar in normalised [-1,1] units.
    Returns NaN if either set is empty.
    """
    if pred_segs.shape[0] == 0 or gt_segs.shape[0] == 0:
        return pred_segs.new_tensor(float("nan"))

    pred_pts = torch.cat([pred_segs[:, :2], pred_segs[:, 2:4]], dim=0)  # [2M, 2]
    gt_pts   = torch.cat([gt_segs[:, :2],   gt_segs[:, 2:4]],   dim=0)  # [2N, 2]

    D = torch.cdist(pred_pts, gt_pts)          # [2M, 2N] Euclidean
    forward  = D.min(dim=1).values.mean()      # each pred endpoint → nearest GT
    backward = D.min(dim=0).values.mean()      # each GT endpoint → nearest pred
    return (forward + backward) / 2.0


def segment_precision_recall_f1(
    pred_segs: torch.Tensor,   # [M, 4]  active predicted
    gt_segs:   torch.Tensor,   # [N, 4]  valid GT
    threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Segment-level precision, recall, and F1 at a distance threshold.

    A predicted segment is a true positive (TP) if it lies within `threshold`
    RMS endpoint distance of any GT segment.
    A GT segment is recalled if any prediction lies within `threshold` of it.

    Args:
        pred_segs:  [M, 4]  active predicted segments
        gt_segs:    [N, 4]  GT segments
        threshold:  τ in normalised coordinate units

    Returns:
        (precision, recall, f1) — scalar tensors on the same device as inputs
    """
    if pred_segs.shape[0] == 0 and gt_segs.shape[0] == 0:
        v = pred_segs.new_tensor(1.0)
        return v, v, v

    device = pred_segs.device if pred_segs.shape[0] > 0 else gt_segs.device
    z = torch.tensor(0.0, device=device)

    if pred_segs.shape[0] == 0 or gt_segs.shape[0] == 0:
        return z, z, z

    D = pi_dist_matrix(pred_segs, gt_segs)     # [M, N]

    precision = (D.min(dim=1).values <= threshold).float().mean()
    recall    = (D.min(dim=0).values <= threshold).float().mean()
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def total_road_length(segs: torch.Tensor) -> float:
    """Sum of Euclidean segment lengths in normalised [-1,1] coords.

    Analogous to maptrace's raster road-pixel fraction: a scalar that captures
    how much road content is present in the image regardless of segment count.

    Args:
        segs: [N, 4]  x1,y1,x2,y2  (valid GT segments only, no padding)

    Returns:
        Scalar float.  Image diagonal in [-1,1] space ≈ 2√2 ≈ 2.83, so a
        value of 0.5 means roughly 18 % of the diagonal is covered by roads.
    """
    if segs is None or segs.shape[0] == 0:
        return 0.0
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    return (dx ** 2 + dy ** 2).sqrt().sum().item()


# Density tier thresholds by valid GT segment count.
# Tuned for densified segments at max_segment_length=0.06 on 512px imagery:
#   super_sparse  < 8   segments  — nearly empty tiles (fields, water)
#   sparse        8–32            — single road or scattered paths
#   medium        33–124          — typical suburban / rural
#   dense         125–332         — urban grid
#   super_dense   ≥ 333           — dense urban / intersection-heavy
_DENSITY_THRESHOLDS = {
    "super_sparse": 8,
    "sparse":       33,
    "medium":       125,
    "dense":        333,
    # super_dense = anything above dense
}
DENSITY_BUCKETS = ("super_sparse", "sparse", "medium", "dense", "super_dense")


def segment_density_bucket(segs: torch.Tensor) -> str:
    """Map a GT segment set to one of five density tier names by segment count.

    Returns one of: ``'super_sparse'``, ``'sparse'``, ``'medium'``,
    ``'dense'``, ``'super_dense'``.
    """
    n = segs.shape[0] if segs is not None else 0
    if n < _DENSITY_THRESHOLDS["super_sparse"]:
        return "super_sparse"
    if n < _DENSITY_THRESHOLDS["sparse"]:
        return "sparse"
    if n < _DENSITY_THRESHOLDS["medium"]:
        return "medium"
    if n < _DENSITY_THRESHOLDS["dense"]:
        return "dense"
    return "super_dense"


def pr_curve(
    pred_list: list[torch.Tensor],   # list of [M_i, 4]
    gt_list:   list[torch.Tensor],   # list of [N_i, 4]
    thresholds: list[float] | None = None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Compute mean precision, recall, and F1 over a range of thresholds.

    Precomputes all pairwise distance matrices once, then sweeps thresholds.

    Returns:
        (thresholds, precisions, recalls, f1s)  — all Python lists of floats
    """
    import numpy as np

    if thresholds is None:
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08,
                      0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

    # Pre-compute pairwise distance matrices on CPU to save GPU memory
    dist_mats = []
    for pred, gt in zip(pred_list, gt_list):
        if pred.shape[0] > 0 and gt.shape[0] > 0:
            dist_mats.append(pi_dist_matrix(pred.cpu(), gt.cpu()))

    if not dist_mats:
        z = [0.0] * len(thresholds)
        return thresholds, z, z, z

    precs, recs, f1s = [], [], []
    for tau in thresholds:
        p_vals, r_vals = [], []
        for D in dist_mats:
            p_vals.append((D.min(dim=1).values <= tau).float().mean().item())
            r_vals.append((D.min(dim=0).values <= tau).float().mean().item())
        p = float(np.mean(p_vals))
        r = float(np.mean(r_vals))
        f = 2 * p * r / (p + r + 1e-8)
        precs.append(p)
        recs.append(r)
        f1s.append(f)

    return thresholds, precs, recs, f1s


# ── Graph-based metrics ────────────────────────────────────────────────────────
# Both APLS and TOPO require building a road-network graph from segments.
# Nodes are segment endpoints snapped to a grid; edges are segment interiors.

def _segs_to_graph(segs: torch.Tensor, snap: float = 0.005):
    """Convert [N,4] segments to a weighted undirected networkx Graph.

    Endpoints within `snap` distance are merged to the same node so that
    nearly-touching roads form a connected graph.  Edge weight = Euclidean
    segment length.

    Returns (G, node_coords) where node_coords is a dict {node_id: (x, y)}.
    """
    import networkx as nx

    if segs is None or segs.shape[0] == 0:
        return nx.Graph(), {}

    pts = segs[:, :2].tolist() + segs[:, 2:4].tolist()   # 2N raw endpoints

    # Union-Find to merge nearby endpoints
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(len(pts)):
        xi, yi = pts[i]
        for j in range(i + 1, len(pts)):
            xj, yj = pts[j]
            if abs(xi - xj) <= snap and abs(yi - yj) <= snap:
                if (xi - xj) ** 2 + (yi - yj) ** 2 <= snap ** 2:
                    union(i, j)

    # Map root → representative coords (mean of members)
    from collections import defaultdict
    clusters: dict = defaultdict(list)
    for i, (x, y) in enumerate(pts):
        clusters[find(i)].append((x, y))

    roots = sorted(clusters.keys())
    root_to_nid = {r: nid for nid, r in enumerate(roots)}
    node_coords = {
        root_to_nid[r]: (
            float(sum(c[0] for c in members) / len(members)),
            float(sum(c[1] for c in members) / len(members)),
        )
        for r, members in clusters.items()
    }

    G = nx.Graph()
    for nid, (x, y) in node_coords.items():
        G.add_node(nid, x=x, y=y)

    n = segs.shape[0]
    for i in range(n):
        x1, y1, x2, y2 = segs[i].tolist()
        u = root_to_nid[find(i)]          # first endpoint index = i
        v = root_to_nid[find(i + n)]      # second endpoint index = i + n
        if u == v:
            continue   # degenerate / self-loop
        w = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        if G.has_edge(u, v):
            G[u][v]["weight"] = min(G[u][v]["weight"], w)
        else:
            G.add_edge(u, v, weight=w)

    return G, node_coords


def apls(
    pred_segs: torch.Tensor,   # [M, 4]
    gt_segs:   torch.Tensor,   # [N, 4]
    n_samples: int = 100,
    snap: float = 0.005,
    seed: int = 0,
) -> float:
    """Average Path Length Similarity (APLS).

    For each sampled pair of connected GT nodes, computes:
        similarity = 1 - |d_gt - d_pred| / d_gt
    where d_gt / d_pred are shortest-path distances in the GT / pred graph.
    Pairs disconnected in pred contribute similarity = 0.
    Pairs disconnected in GT are skipped.

    Returns a scalar in [0, 1].  Returns NaN if no valid pairs found.

    Reference: Van Etten et al. "SpaceNet: A Remote Sensing Dataset and
    Challenge Series" (2019).
    """
    import networkx as nx
    import random as rng_mod

    if gt_segs.shape[0] == 0 or pred_segs.shape[0] == 0:
        return float("nan")

    gt_G,   gt_nodes   = _segs_to_graph(gt_segs,   snap=snap)
    pred_G, pred_nodes = _segs_to_graph(pred_segs, snap=snap)

    gt_node_list = list(gt_G.nodes())
    if len(gt_node_list) < 2:
        return float("nan")

    rng = rng_mod.Random(seed)
    pairs = []
    attempts = 0
    while len(pairs) < n_samples and attempts < n_samples * 20:
        u, v = rng.sample(gt_node_list, 2)
        if nx.has_path(gt_G, u, v):
            pairs.append((u, v))
        attempts += 1

    if not pairs:
        return float("nan")

    # Find the nearest pred node for each GT node (by coordinate)
    pred_xy = torch.tensor(
        [[pred_nodes[nid][0], pred_nodes[nid][1]] for nid in sorted(pred_nodes)],
        dtype=torch.float32,
    ) if pred_nodes else None

    def nearest_pred_node(gt_nid):
        if pred_xy is None or pred_xy.shape[0] == 0:
            return None
        gx, gy = gt_nodes[gt_nid]
        dists = ((pred_xy[:, 0] - gx) ** 2 + (pred_xy[:, 1] - gy) ** 2)
        idx = int(dists.argmin())
        return sorted(pred_nodes.keys())[idx]

    scores = []
    for u, v in pairs:
        d_gt = nx.shortest_path_length(gt_G, u, v, weight="weight")
        if d_gt <= 0:
            continue
        pu = nearest_pred_node(u)
        pv = nearest_pred_node(v)
        if pu is None or pv is None or not nx.has_path(pred_G, pu, pv):
            scores.append(0.0)
        else:
            d_pred = nx.shortest_path_length(pred_G, pu, pv, weight="weight")
            scores.append(max(0.0, 1.0 - abs(d_gt - d_pred) / d_gt))

    return float(sum(scores) / len(scores)) if scores else float("nan")


def topo(
    pred_segs: torch.Tensor,   # [M, 4]
    gt_segs:   torch.Tensor,   # [N, 4]
    n_samples: int = 100,
    snap: float = 0.005,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Topology metric: path connectivity precision, recall, and F1.

    Samples node pairs from GT.  For each pair:
    - Recall: GT-connected pairs that are also connected in pred graph.
    - Precision: GT-connected pairs for which pred is connected AND there
      exists a pred-path that maps back to a connected GT path (i.e., pred
      doesn't hallucinate spurious cross-connections beyond a distance gate).

    Returns (precision, recall, f1) in [0, 1].  Returns (nan, nan, nan)
    if no valid pairs.

    Reference: Wegner et al. "Road Networks and Connectivity" (2013);
    also used in Sat2Graph evaluation.
    """
    import networkx as nx
    import random as rng_mod

    if gt_segs.shape[0] == 0 or pred_segs.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")

    gt_G,   gt_nodes   = _segs_to_graph(gt_segs,   snap=snap)
    pred_G, pred_nodes = _segs_to_graph(pred_segs, snap=snap)

    gt_node_list = list(gt_G.nodes())
    pred_node_list = list(pred_G.nodes()) if pred_G.number_of_nodes() > 0 else []

    if len(gt_node_list) < 2:
        return float("nan"), float("nan"), float("nan")

    pred_xy = torch.tensor(
        [[pred_nodes[nid][0], pred_nodes[nid][1]] for nid in sorted(pred_nodes)],
        dtype=torch.float32,
    ) if pred_nodes else None
    pred_sorted_keys = sorted(pred_nodes.keys()) if pred_nodes else []

    def nearest_pred(gx, gy):
        if pred_xy is None or pred_xy.shape[0] == 0:
            return None
        dists = (pred_xy[:, 0] - gx) ** 2 + (pred_xy[:, 1] - gy) ** 2
        return pred_sorted_keys[int(dists.argmin())]

    rng = rng_mod.Random(seed)
    pairs = []
    attempts = 0
    while len(pairs) < n_samples and attempts < n_samples * 20:
        u, v = rng.sample(gt_node_list, 2)
        if nx.has_path(gt_G, u, v):
            pairs.append((u, v))
        attempts += 1

    if not pairs:
        return float("nan"), float("nan"), float("nan")

    recall_hits = 0
    for u, v in pairs:
        pu = nearest_pred(gt_nodes[u][0], gt_nodes[u][1])
        pv = nearest_pred(gt_nodes[v][0], gt_nodes[v][1])
        if pu is not None and pv is not None and nx.has_path(pred_G, pu, pv):
            recall_hits += 1

    # Precision: sample pred-connected pairs, check they're also GT-connected
    gt_xy = torch.tensor(
        [[gt_nodes[nid][0], gt_nodes[nid][1]] for nid in sorted(gt_nodes)],
        dtype=torch.float32,
    )
    gt_sorted_keys = sorted(gt_nodes.keys())

    def nearest_gt(px, py):
        dists = (gt_xy[:, 0] - px) ** 2 + (gt_xy[:, 1] - py) ** 2
        return gt_sorted_keys[int(dists.argmin())]

    pred_pairs = []
    attempts = 0
    while len(pred_pairs) < n_samples and attempts < n_samples * 20 and len(pred_node_list) >= 2:
        pu, pv = rng.sample(pred_node_list, 2)
        if nx.has_path(pred_G, pu, pv):
            pred_pairs.append((pu, pv))
        attempts += 1

    prec_hits = 0
    for pu, pv in pred_pairs:
        gu = nearest_gt(pred_nodes[pu][0], pred_nodes[pu][1])
        gv = nearest_gt(pred_nodes[pv][0], pred_nodes[pv][1])
        if nx.has_path(gt_G, gu, gv):
            prec_hits += 1

    rec  = recall_hits / len(pairs) if pairs else float("nan")
    prec = prec_hits / len(pred_pairs) if pred_pairs else float("nan")
    if prec != prec or rec != rec:   # nan check
        return float("nan"), float("nan"), float("nan")
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1
