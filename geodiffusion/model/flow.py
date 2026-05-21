"""
Conditional Flow Matching (CFM) with straight-line (OT) paths.

Source x₀ ~ anchor distribution (spoke-wheel or Gaussian).
Target x₁ ~ GT road segments (after matching).

Linear interpolation path:
    x_t = (1 - t) · x₀ + t · x₁        t ∈ [0, 1]

Constant velocity (gradient of the path):
    v = dx_t/dt = x₁ - x₀

The model is trained to predict v from (x_t, t, image).
At inference, integrate from t=0 → t=1 via Euler:
    x_{t+Δ} = x_t + Δ · v_θ(x_t, t, image)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FlowMatching(nn.Module):
    """Utility module for straight-line (OT) conditional flow matching.

    Registered as an nn.Module so it can be included in checkpoints cleanly,
    but it has no learnable parameters.
    """

    def interpolate(
        self,
        x0: torch.Tensor,  # [B, N, 5]  source (anchors)
        x1: torch.Tensor,  # [B, N, 5]  target
        t:  torch.Tensor,  # [B]        time in [0, 1]
    ) -> torch.Tensor:
        """Sample a point on the straight-line path at time t.

        Returns:
            x_t: [B, N, 5]
        """
        t_ = t.view(-1, 1, 1)       # broadcast over N and 5
        return (1.0 - t_) * x0 + t_ * x1

    def velocity(
        self,
        x0: torch.Tensor,  # [B, N, 5]
        x1: torch.Tensor,  # [B, N, 5]
    ) -> torch.Tensor:
        """Ground-truth constant velocity along the straight-line path.

        Returns:
            v: [B, N, 5]  — same at every t for OT paths
        """
        return x1 - x0

    @torch.no_grad()
    def euler_integrate(
        self,
        x0: torch.Tensor,           # [B, N, 5]
        model: nn.Module,
        image: torch.Tensor | None,  # [B, 3, H, W]
        steps: int = 10,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Integrate from t=0 to t=1 using Euler steps.

        Args:
            x0:     starting positions (anchors)
            model:  velocity field v_θ(x_t, t, image)
            image:  conditioning image (float, [0,1] normalised)
            steps:  number of Euler integration steps

        Returns:
            x1_pred: [B, N, 5]  predicted final positions
        """
        dt = 1.0 / steps
        B = x0.shape[0]
        xt = x0.clone()
        if device is None:
            device = x0.device

        for i in range(steps):
            t_val = i * dt
            t_tensor = torch.full((B,), t_val, dtype=torch.float32, device=device)
            v_pred = model(xt, t_tensor, image=image)
            xt = xt + dt * v_pred
            xt[..., :4] = xt[..., :4].clamp(-1.0, 1.0)  # keep coords in image bounds
        return xt


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian fallback (backward-compatible initialisation)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianAnchorInit:
    """Generates pure-Gaussian x₀ as a drop-in for SpokeWheelAnchors.

    Used when cfg.anchors.type == "gaussian" to allow switching via Hydra.
    """

    def __init__(self, n_anchors: int):
        self.n_anchors = n_anchors

    @property
    def n(self) -> int:
        return self.n_anchors

    def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return [B, N_anchors, 5] standard-normal samples."""
        return torch.randn(batch_size, self.n_anchors, 5, device=device)
