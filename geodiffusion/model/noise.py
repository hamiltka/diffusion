import torch
import torch.nn as nn
from .schedules import build_schedule


class GaussianNoise(nn.Module):
    """Forward diffusion process: adds Gaussian noise to vector road segments.

    Registered buffers are automatically moved to the correct device by Lightning.
    """

    def __init__(self, timesteps: int, schedule_type: str = "cosine"):
        super().__init__()
        self.timesteps = timesteps

        alphas_cumprod, a_sqrt, a_sqrt_one_minus = build_schedule(schedule_type, timesteps)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alpha", a_sqrt)
        self.register_buffer("sqrt_one_minus_alpha", a_sqrt_one_minus)

    def add_noise(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0).

        Args:
            x:     [B, N, 5] clean segments
            t:     [B]       timestep indices
            noise: [B, N, 5] optional pre-sampled noise

        Returns:
            [B, N, 5] noisy segments
        """
        if noise is None:
            noise = torch.randn_like(x)

        # Gather per-timestep scale factors and reshape for broadcast
        sqrt_a = self.sqrt_alpha[t]            # [B]
        sqrt_1ma = self.sqrt_one_minus_alpha[t]  # [B]
        for _ in range(x.dim() - 1):
            sqrt_a = sqrt_a.unsqueeze(-1)
            sqrt_1ma = sqrt_1ma.unsqueeze(-1)

        return sqrt_a * x + sqrt_1ma * noise

    # Alias used elsewhere in the codebase
    q_sample = add_noise
