import torch


def linear_alpha_schedule(timesteps):
    """Linear schedule: alpha decays from 1.0 → 1e-5."""
    alpha_start, alpha_end = 1.0, 1e-5
    t = torch.linspace(0, 1, timesteps, dtype=torch.float32)
    per_step = alpha_start + t * (alpha_end - alpha_start)
    alphas_cumprod = torch.cumprod(per_step, dim=0)
    return alphas_cumprod, torch.sqrt(alphas_cumprod), torch.sqrt(1.0 - alphas_cumprod)


def cosine_alpha_schedule(timesteps, s=0.008):
    """Cosine schedule (Nichol & Dhariwal 2021)."""
    assert s >= 0
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas_cumprod = alphas_cumprod[1:]
    return alphas_cumprod, torch.sqrt(alphas_cumprod), torch.sqrt(1.0 - alphas_cumprod)


def sqrt_alpha_schedule(timesteps):
    """Square-root schedule: ᾱ_t = 1 - sqrt(t/T)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
    alphas_cumprod = 1.0 - torch.sqrt(x)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas_cumprod = alphas_cumprod[1:]
    return alphas_cumprod, torch.sqrt(alphas_cumprod), torch.sqrt(1.0 - alphas_cumprod)


def sigmoid_alpha_schedule(timesteps, start=-3, end=3, tau=1.0):
    """Sigmoid-based schedule: S-shaped noise progression."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
    alphas_cumprod = torch.sigmoid(-(x * (end - start) + start) / tau)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas_cumprod = alphas_cumprod[1:]
    return alphas_cumprod, torch.sqrt(alphas_cumprod), torch.sqrt(1.0 - alphas_cumprod)


_SCHEDULES = {
    "linear": linear_alpha_schedule,
    "cosine": cosine_alpha_schedule,
    "sqrt": sqrt_alpha_schedule,
    "sigmoid": sigmoid_alpha_schedule,
}


def build_schedule(schedule_type: str, timesteps: int):
    """Return (alphas_cumprod, sqrt_alpha, sqrt_one_minus_alpha) for the given schedule."""
    if schedule_type not in _SCHEDULES:
        raise ValueError(f"Unknown schedule_type '{schedule_type}'. Choose from {list(_SCHEDULES)}")
    return _SCHEDULES[schedule_type](timesteps)
