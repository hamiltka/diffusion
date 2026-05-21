"""
Standalone test for all tensorboard visualization grids.
Run with: conda run -n trace_geo python test_viz.py
Saves PNGs to /tmp/test_viz_*.png for inspection.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ── constants (same as lightning_module) ─────────────────────────────────────
_GT_COLOR   = "#00FFFF"
_PRED_COLOR = "#FF00FF"
_ANC_COLOR  = "#95a5a6"
_ERR_GOOD   = "#00FFFF"
_ERR_MED    = "#FFFF00"
_ERR_BAD    = "#FF00FF"

def _draw_segs(ax, segs, color, lw=2.5, alpha=0.85, img_size=512):
    if segs is None or len(segs) == 0:
        return
    if torch.is_tensor(segs):
        segs = segs.tolist()
    for x1, y1, x2, y2 in segs:
        xs = [(x1+1)/2 * img_size, (x2+1)/2 * img_size]
        ys = [(y1+1)/2 * img_size, (y2+1)/2 * img_size]
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, solid_capstyle='round')
        ax.scatter(xs, ys, color=color, s=8, alpha=alpha, zorder=5, linewidths=0)

def pi_dist_matrix(a, b):
    """dummy L2 distance for testing"""
    # a: [M,4], b: [K,4] -> [M,K]
    a = a.unsqueeze(1)  # [M,1,4]
    b = b.unsqueeze(0)  # [1,K,4]
    return ((a - b)**2).sum(-1).sqrt()

# ── dummy data ────────────────────────────────────────────────────────────────
def make_dummy_img():
    img = np.random.rand(512, 512, 3) * 0.5 + 0.25
    # add some green patches
    img[100:200, 100:200, 1] += 0.3
    return np.clip(img, 0, 1)

def make_dummy_segs(n=20, active_frac=0.4):
    """Return [n, 5] tensor with (x1,y1,x2,y2, score) in [-1,1]"""
    segs = torch.rand(n, 5) * 2 - 1  # all in [-1,1]
    segs[:, 4] = torch.randn(n)       # active scores
    return segs

def make_viz_data(n_rows=4):
    rows = []
    for _ in range(n_rows):
        xp = make_dummy_segs(40)
        x0 = make_dummy_segs(40)
        gt = torch.rand(15, 4) * 2 - 1
        rows.append({
            "img_np":   make_dummy_img(),
            "x1pred_b": xp,
            "x0_b":     x0,
            "gt_segs":  gt,
        })
    return rows

# ══════════════════════════════════════════════════════════════════════════════
# 1. qual_grid
# ══════════════════════════════════════════════════════════════════════════════
def make_qual_grid(viz_data):
    n = len(viz_data)
    ncols = 5
    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 3.5, n * 3.5), dpi=120)
    if n == 1:
        axes = axes[None, :]

    col_titles = ["Satellite", "Ground Truth", "Predicted (active>0)",
                  "Active confidence", "Error map"]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=9, pad=4)

    cmap_conf = cm.RdYlGn

    for row, data in enumerate(viz_data):
        img     = np.clip(data["img_np"], 0, 1)
        x1pred  = data["x1pred_b"]
        gt_segs = data["gt_segs"]

        pred_active = x1pred[x1pred[:, 4] > 0.0, :4]
        pred_all    = x1pred[:, :4]
        act_scores  = x1pred[:, 4]

        if pred_active.shape[0] > 0 and gt_segs.shape[0] > 0:
            D_err = pi_dist_matrix(pred_active, gt_segs)
            min_dists = D_err.min(dim=1).values
        else:
            min_dists = pred_active.new_tensor([])

        for col in range(ncols):
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis("off")

        _draw_segs(axes[row, 1], gt_segs, color=_GT_COLOR, lw=2.5, alpha=0.9)
        _draw_segs(axes[row, 2], pred_active, color=_PRED_COLOR, lw=2.5, alpha=0.9)

        if pred_all.shape[0] > 0:
            norm_scores = ((act_scores.clamp(-1, 1) + 1) / 2.0).numpy()
            order = np.argsort(norm_scores)
            for i in order:
                x1, y1, x2, y2 = pred_all[i].tolist()
                score = norm_scores[i]
                rgba  = cmap_conf(score)
                alpha = 0.3 + 0.5 * score
                axes[row, 3].plot(
                    [(x1+1)/2*512, (x2+1)/2*512], [(y1+1)/2*512, (y2+1)/2*512],
                    color=rgba, linewidth=0.5, alpha=alpha,
                )

        if pred_active.shape[0] > 0 and min_dists.shape[0] > 0:
            for i in range(pred_active.shape[0]):
                d = min_dists[i].item()
                color = _ERR_GOOD if d <= 0.05 else (_ERR_MED if d <= 0.10 else _ERR_BAD)
                x1, y1, x2, y2 = pred_active[i].tolist()
                axes[row, 4].plot(
                    [(x1+1)/2*512, (x2+1)/2*512], [(y1+1)/2*512, (y2+1)/2*512],
                    color=color, linewidth=2.5, alpha=0.9, solid_capstyle='round',
                )

        axes[row, 0].set_ylabel(f"Sample {row}", fontsize=8)

    legend_elements = [
        Line2D([0], [0], color=_ERR_GOOD, lw=2, label="dist ≤ 0.05"),
        Line2D([0], [0], color=_ERR_MED,  lw=2, label="0.05–0.10"),
        Line2D([0], [0], color=_ERR_BAD,  lw=2, label="dist > 0.10"),
    ]
    axes[-1, 4].legend(handles=legend_elements, loc="lower right", fontsize=6, framealpha=0.6)

    sm = cm.ScalarMappable(cmap=cmap_conf, norm=mcolors.Normalize(-1, 1))
    sm.set_array([])

    fig.suptitle("TEST — Qualitative results", fontsize=11, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.88, 0.97])
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
    fig.colorbar(sm, cax=cbar_ax, label="active score")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 2. flow_trajectory
# ══════════════════════════════════════════════════════════════════════════════
def make_flow_trajectory(viz_data):
    n = len(viz_data)
    T_VEC = [0.0, 0.25, 0.50, 0.75, 1.0]
    ncols = len(T_VEC) + 1

    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 3.0, n * 3.0), dpi=120)
    if n == 1:
        axes = axes[None, :]

    col_labels = [f"t = {t:.2f}" for t in T_VEC] + ["GT"]
    for c, lbl in enumerate(col_labels):
        axes[0, c].set_title(lbl, fontsize=9, pad=4)

    cmap_t = cm.plasma

    for row, data in enumerate(viz_data):
        img    = np.clip(data["img_np"], 0, 1)
        x0_b   = data["x0_b"]
        xp_b   = data["x1pred_b"]
        gt_segs = data["gt_segs"]

        active_mask = xp_b[:, 4] > 0.0
        if not active_mask.any():
            k = min(20, xp_b.shape[0])
            top_idx = xp_b[:, 4].topk(k).indices
            active_mask = torch.zeros(xp_b.shape[0], dtype=torch.bool)
            active_mask[top_idx] = True
        x0_active = x0_b[active_mask, :4]
        xp_active = xp_b[active_mask, :4]

        for col, t_val in enumerate(T_VEC):
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis("off")
            if x0_active.shape[0] > 0:
                xt = (1.0 - t_val) * x0_active + t_val * xp_active
                rgba = cmap_t(t_val)
                _draw_segs(ax, xt, color=rgba, lw=2.0, alpha=0.85)

        axes[row, ncols-1].imshow(img)
        axes[row, ncols-1].axis("off")
        _draw_segs(axes[row, ncols-1], gt_segs, color=_GT_COLOR, lw=2.5, alpha=0.9)
        axes[row, 0].set_ylabel(f"Sample {row}", fontsize=8)

    sm = cm.ScalarMappable(cmap=cmap_t, norm=mcolors.Normalize(0, 1))
    sm.set_array([])

    fig.suptitle("TEST — Flow trajectory (active anchors)", fontsize=11, y=1.01)
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    cbar_ax = fig.add_axes([0.10, 0.025, 0.80, 0.025])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label="flow time t")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 3. anchor_flow_grid
# ══════════════════════════════════════════════════════════════════════════════
def make_anchor_flow_grid(viz_data):
    n = len(viz_data)
    fig, axes = plt.subplots(n, 4, figsize=(4 * 3.5, n * 3.5), dpi=120)
    if n == 1:
        axes = axes[None, :]

    col_titles = ["t=0  (spoke-wheel)", "t=1  (all, by score)", "Top-30 highest score", "GT"]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=9, pad=4)

    cmap_score = cm.RdYlGn

    for row, data in enumerate(viz_data):
        img     = np.clip(data["img_np"], 0, 1)
        x0_b    = data["x0_b"]
        xp_b    = data["x1pred_b"]
        gt_segs = data["gt_segs"]

        scores  = xp_b[:, 4].numpy()
        norm_sc = (np.clip(scores, -1, 1) + 1) / 2.0

        for col in range(4):
            axes[row, col].imshow(img)
            axes[row, col].axis("off")

        for i in range(x0_b.shape[0]):
            x1, y1, x2, y2 = x0_b[i, :4].tolist()
            axes[row, 0].plot(
                [(x1+1)/2*512, (x2+1)/2*512], [(y1+1)/2*512, (y2+1)/2*512],
                color="#888888", linewidth=0.3, alpha=0.45,
            )

        order = np.argsort(norm_sc)
        for i in order:
            x1, y1, x2, y2 = xp_b[i, :4].tolist()
            rgba  = cmap_score(norm_sc[i])
            alpha = 0.2 + 0.7 * norm_sc[i]
            axes[row, 1].plot(
                [(x1+1)/2*512, (x2+1)/2*512], [(y1+1)/2*512, (y2+1)/2*512],
                color=rgba, linewidth=0.45, alpha=alpha,
            )

        top_k   = min(30, xp_b.shape[0])
        top_idx = np.argsort(scores)[-top_k:]
        for i in top_idx:
            x1, y1, x2, y2 = xp_b[i, :4].tolist()
            rgba = cmap_score(norm_sc[i])
            axes[row, 2].plot(
                [(x1+1)/2*512, (x2+1)/2*512], [(y1+1)/2*512, (y2+1)/2*512],
                color=rgba, linewidth=2.5, alpha=0.9, solid_capstyle='round',
            )

        _draw_segs(axes[row, 3], gt_segs, color=_GT_COLOR, lw=2.5, alpha=0.9)
        axes[row, 0].set_ylabel(f"Sample {row}", fontsize=8)

    sm = cm.ScalarMappable(cmap=cmap_score, norm=mcolors.Normalize(-1, 1))
    sm.set_array([])

    fig.suptitle("TEST — Deconstruction → Reconstruction", fontsize=10, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.88, 0.97])
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
    fig.colorbar(sm, cax=cbar_ax, label="active score  (red=inactive  →  green=active)")
    return fig

# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    viz_data = make_viz_data(n_rows=4)

    print("Generating qual_grid...")
    fig = make_qual_grid(viz_data)
    fig.savefig("/tmp/test_viz_qual_grid.png", bbox_inches="tight", dpi=100)
    plt.close(fig)

    print("Generating flow_trajectory...")
    fig = make_flow_trajectory(viz_data[:2])
    fig.savefig("/tmp/test_viz_flow_trajectory.png", bbox_inches="tight", dpi=100)
    plt.close(fig)

    print("Generating anchor_flow_grid...")
    fig = make_anchor_flow_grid(viz_data)
    fig.savefig("/tmp/test_viz_anchor_flow_grid.png", bbox_inches="tight", dpi=100)
    plt.close(fig)

    print("Done. Files saved to /tmp/test_viz_*.png")
