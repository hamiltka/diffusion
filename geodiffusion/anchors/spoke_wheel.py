"""
Spoke-wheel anchor (source) distribution for flow matching.

Divides [-1,1]² into a regular grid.  At each cell centre, N evenly-spaced
spokes are placed, forming short line segments that point outward in different
directions.  These act as the source distribution x₀ that the flow-matching
model deforms towards the ground-truth road network x₁.

Geometry (all coordinates in normalised [-1, 1] space):

    cell centre c = (cx, cy)
    spoke angle   θ_k = k * 2π / n_spokes,  k = 0…n_spokes-1
    anchor p1     = c
    anchor p2     = p1 + spoke_length * (cos θ_k, sin θ_k)

Config keys (under cfg.anchors):
    grid_size       int   — cells per side (grid_size² cells total)
    n_spokes        int   — spokes per cell
    spoke_length    float — half-length in [-1,1] coords  (≈ spoke_length/2 × img_px)
"""
from __future__ import annotations

import math

import torch
from omegaconf import DictConfig


class SpokeWheelAnchors:
    """Vectorised spoke-wheel anchor generator.

    Produces [B, N_anchors, 5] tensors where the 5 channels are
    (x1, y1, x2, y2, active) in [-1, 1] normalised image coordinates.

    Each anchor is a short line segment (p1 → p2).  At init time the full
    grid geometry is pre-computed once and stored as CPU tensors; they are
    moved to the target device on demand inside generate().

    Two generation modes are supported:

      grid mode  (default, used at inference and in most training runs)
        A deterministic grid of spokes covering the entire image uniformly.
        N_anchors = grid_size² × n_spokes  (e.g. 8×8×24 = 1536).
        No GT information is used — the model must learn to map every anchor
        to either a road segment or "inactive" purely from the image.

      gt_seeded mode  (cfg.anchors.mode = "gt_seeded")
        Anchors are initialised near GT segments (+ small Gaussian noise).
        Useful for map-correction tasks where noisy prior segments are given
        and the model refines them using the satellite image.
    """

    def __init__(self, cfg: DictConfig):
        self.grid_size: int = int(cfg.grid_size)
        self.n_spokes: int = int(cfg.n_spokes)
        self.spoke_length: float = float(cfg.spoke_length)

        # Total anchors per image in grid mode
        self.n_anchors: int = self.grid_size ** 2 * self.n_spokes

        # Pre-compute deterministic cell centres and spoke angles once.
        # Both are kept on CPU and moved to the target device inside generate().
        g = self.grid_size
        cell = 2.0 / g                                            # cell width in [-1,1] space
        idxs = torch.arange(g, dtype=torch.float32)
        cx = -1.0 + cell * (idxs + 0.5)                          # centre coords along one axis [g]
        # Build all g² cell centres via meshgrid
        gx, gy = torch.meshgrid(cx, cx, indexing="ij")
        self._cell_cx: torch.Tensor = gx.reshape(-1)             # [g²]
        self._cell_cy: torch.Tensor = gy.reshape(-1)             # [g²]

        # n_spokes evenly-spaced angles covering the full circle [0, 2π)
        angles = torch.arange(self.n_spokes, dtype=torch.float32) * (2 * math.pi / self.n_spokes)
        self._base_angles: torch.Tensor = angles                  # [n_spokes]

    # ------------------------------------------------------------------
    def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return the fixed spoke-wheel source tensor.

        Returns:
            anchors: [B, N_anchors, 5]  —  (x1, y1, x2, y2, active)
        """
        B = batch_size
        n_cells = self.grid_size ** 2
        K = self.n_spokes
        N = self.n_anchors           # n_cells × K

        # Cell centres repeated over spokes: [n_cells*K]
        cx = self._cell_cx.to(device).repeat_interleave(K)       # [N]
        cy = self._cell_cy.to(device).repeat_interleave(K)

        # Spoke angles tiled over cells: [N]
        angles = self._base_angles.to(device).repeat(n_cells)    # [N]

        # Spoke root: p1 = cell_centre
        p1x = cx[None].expand(B, -1)           # [B, N]
        p1y = cy[None].expand(B, -1)

        # Spoke tip: p2 = p1 + spoke_length * unit_direction
        theta = angles[None].expand(B, -1)     # [B, N]
        p2x   = (p1x + self.spoke_length * torch.cos(theta)).clamp(-1.0, 1.0)
        p2y   = (p1y + self.spoke_length * torch.sin(theta)).clamp(-1.0, 1.0)

        # Active: initialised to 0
        active = torch.zeros(B, N, device=device)

        anchors = torch.stack([p1x, p1y, p2x, p2y, active], dim=-1)  # [B, N, 5]
        return anchors

    # ------------------------------------------------------------------
    def generate_from_gt(
        self,
        gt_segs: torch.Tensor,       # [B, N_gt, 4]  (x1,y1,x2,y2) in [-1,1]
        invalid_mask: torch.Tensor,  # [B, N_gt]  True = padded/invalid slot
    ) -> torch.Tensor:               # [B, N_anchors, 5]
        """GT-seeded anchor generator: full spoke grid with matched spokes flagged active.

        Returns the full [B, N_anchors, 5] spoke-wheel grid, just like generate(),
        except:
          - Each valid GT segment is assigned to the spoke within the nearest grid
            cell whose tip is closest (permutation-invariant) to the GT endpoints.
          - All other spokes remain at their grid positions with active = -1.

        Selecting the closest spoke tip aligns this assignment with the Hungarian
        matching performed in build_targets, so the target active velocity is
        approximately zero (no reclassification needed) and gradients flow cleanly
        to the coordinate channels.
        """
        B, N_gt, _ = gt_segs.shape
        device = gt_segs.device

        # Full grid for all batch items, all inactive to start.
        x0 = self.generate(B, device)   # [B, N_anchors, 5],  active col = 0
        x0[:, :, 4] = -1.0              # all spokes inactive

        # Midpoints of GT segments: [B, N_gt, 2]
        gt_mid = (gt_segs[..., :2] + gt_segs[..., 2:]) * 0.5

        K = self.n_spokes
        n_cells = self.grid_size ** 2
        cell_centres = torch.stack(
            [self._cell_cx.to(device), self._cell_cy.to(device)], dim=-1
        )  # [n_cells, 2]

        # Find the nearest cell for each GT segment.
        cell_diff    = gt_mid.unsqueeze(2) - cell_centres[None, None]   # [B, N_gt, n_cells, 2]
        nearest_cell = cell_diff.pow(2).sum(-1).argmin(-1)              # [B, N_gt]

        # Compute spoke tip positions for every cell: [n_cells, K, 2]
        angles = self._base_angles.to(device)                           # [K]
        tip_x = (self._cell_cx.to(device)[:, None]
                 + self.spoke_length * torch.cos(angles)[None, :])      # [n_cells, K]
        tip_y = (self._cell_cy.to(device)[:, None]
                 + self.spoke_length * torch.sin(angles)[None, :])
        spoke_tips = torch.stack([tip_x, tip_y], dim=-1)                # [n_cells, K, 2]

        # For each GT segment, select the spoke in its nearest cell whose tip
        # is closest to either GT endpoint (permutation-invariant, same metric
        # as build_targets).  Since all K spokes in a cell share the same p1
        # (cell centre), only the tip distance differs across k.
        #
        # near_tips: [B, N_gt, K, 2]
        near_tips = spoke_tips[nearest_cell]                            # [B, N_gt, K, 2]
        gt_p1 = gt_segs[..., :2].unsqueeze(2)                          # [B, N_gt, 1, 2]
        gt_p2 = gt_segs[..., 2:4].unsqueeze(2)
        d_p1 = (near_tips - gt_p1).pow(2).sum(-1)                      # [B, N_gt, K]
        d_p2 = (near_tips - gt_p2).pow(2).sum(-1)
        best_k   = torch.minimum(d_p1, d_p2).argmin(-1)                # [B, N_gt]
        nearest  = nearest_cell * K + best_k                           # [B, N_gt]

        # Mark matched spokes active=+1.
        for b in range(B):
            valid_idx = (~invalid_mask[b]).nonzero(as_tuple=True)[0]  # [M]
            matched   = nearest[b, valid_idx]                         # [M] spoke indices
            x0[b, matched, 4] = 1.0

        return x0   # [B, N_anchors, 5]

    def n(self) -> int:
        """Total number of anchors per image."""
        return self.n_anchors


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "grid_size": 8,
        "n_spokes": 24,
        "spoke_length": 0.08,
        "mode": "grid",
    })
    sw = SpokeWheelAnchors(cfg)
    device = torch.device("cpu")

    print("=== test generate() ===")
    anchors = sw.generate(batch_size=2, device=device)
    assert anchors.shape == (2, sw.n_anchors, 5), f"Bad shape: {anchors.shape}"
    assert anchors[..., :4].abs().max() <= 1.0, "Coords out of [-1,1]"
    assert (anchors[..., 4] == 0).all(), "Active channel should be zero at init"
    print(f"  n_anchors = {sw.n_anchors}  (8×8×24 = {8*8*24})")
    print(f"  shape     = {tuple(anchors.shape)}")
    print(f"  coord range: [{anchors[..., :4].min():.3f}, {anchors[..., :4].max():.3f}]")
    print("  PASSED")

    print("\n=== test generate_from_gt() ===")
    B, N_gt = 2, 50
    gt = torch.rand(B, N_gt, 4) * 2 - 1          # random segs in [-1,1]
    invalid = torch.zeros(B, N_gt, dtype=torch.bool)
    invalid[:, -5:] = True                        # last 5 slots are padded
    x0 = sw.generate_from_gt(gt, invalid)
    assert x0.shape == (B, sw.n_anchors, 5), f"Bad shape: {x0.shape}"
    assert x0[..., :4].abs().max() <= 1.0, "Coords out of [-1,1]"
    # Active channel: matched spokes = +1, rest = -1
    assert (x0[..., 4].abs() == 1.0).all(), "Active channel must be ±1"

    # Each returned segment must be an exact spoke from the grid
    grid = sw.generate(1, device)[0]  # [N_anchors, 5]
    spoke_mids = torch.stack([(grid[:, 0]+grid[:, 2])*0.5, (grid[:, 1]+grid[:, 3])*0.5], -1)
    x0_mids = torch.stack([(x0[0, :, 0]+x0[0, :, 2])*0.5, (x0[0, :, 1]+x0[0, :, 3])*0.5], -1)
    # Every x0 midpoint must be one of the spoke midpoints (no jitter, so exact match)
    min_dists = ((x0_mids.unsqueeze(1) - spoke_mids.unsqueeze(0)).pow(2).sum(-1)).min(-1).values
    assert min_dists.max() < 1e-5, f"x0 not from spoke grid! max dist = {min_dists.max():.6f}"
    n_active = (x0[0, :, 4] == 1.0).sum().item()

    print(f"  shape     = {tuple(x0.shape)}")
    print(f"  active spokes (batch 0): {n_active} / {sw.n_anchors}")
    print(f"  all anchors are exact spokes: max grid dist = {min_dists.max():.2e}")
    print("  PASSED")

    print("\n=== visualise generate() — saving spoke_wheel_test.png ===")
    a = anchors[0]   # [N, 5]  — first image in batch
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal"); ax.set_title(f"SpokeWheelAnchors  grid={cfg.grid_size}  spokes={cfg.n_spokes}  len={cfg.spoke_length}")
    for seg in a:
        x1, y1, x2, y2 = seg[:4].tolist()
        ax.plot([x1, x2], [y1, y2], color="steelblue", lw=0.6, alpha=0.7)
    ax.scatter(a[:, 0], a[:, 1], s=2, color="red", zorder=3)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig("spoke_wheel_test.png", dpi=120)
    plt.close(fig)
    print("  Saved spoke_wheel_test.png")

    print("\n=== visualise generate_from_gt() — saving spoke_wheel_gtseeded_test.png ===")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for bi, ax in enumerate(axes):
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal")
        ax.set_title(f"gt_seeded  batch={bi}")
        gt_b    = gt[bi][~invalid[bi]].numpy()
        active_mask = x0[bi, :, 4] == 1.0
        x0_active   = x0[bi][active_mask].numpy()
        x0_inactive = x0[bi][~active_mask].numpy()
        for s in gt_b:
            ax.plot([s[0], s[2]], [s[1], s[3]], color="cyan", lw=1.2, alpha=0.7)
        for s in x0_inactive:
            ax.plot([s[0], s[2]], [s[1], s[3]], color="#95a5a6", lw=0.4, alpha=0.4)
        for s in x0_active:
            ax.plot([s[0], s[2]], [s[1], s[3]], color="magenta", lw=1.0, alpha=0.8)
        ax.legend(handles=[
            mlines.Line2D([], [], color="cyan",    label="GT"),
            mlines.Line2D([], [], color="magenta", label="Noisy anchor"),
        ], fontsize=7)
    fig.tight_layout()
    fig.savefig("spoke_wheel_gtseeded_test.png", dpi=120)
    plt.close(fig)
    print("  Saved spoke_wheel_gtseeded_test.png")

    print("\nAll tests passed.")
