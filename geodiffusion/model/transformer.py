"""
Transformer-based velocity-field model for vector road-network flow matching.

Architecture:
  - Per-segment linear embedding  (5 → 512)
  - Learnable positional embeddings
  - Continuous timestep embedding  (t ∈ [0,1] → sinusoidal → 512)
  - 8 × CrossAttentionBlock (self-attn on segments + cross-attn to image features)
  - DeepLabV3-ResNet101 image encoder
  - Single velocity prediction head  (512 → 5)

The model predicts the flow velocity v_θ(x_t, t, image) ≈ x₁ − x₀ for all
5 channels: (x1, y1, x2, y2, active).  There is no separate mask head —
the active channel is just the 5th velocity component.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Timestep embedding
# ─────────────────────────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """Continuous sinusoidal timestep embedding for t ∈ [0, 1].

    Scales t by `scale` (default 1000) before computing frequencies so the
    embedding has fine granularity over the [0, 1] range.
    """

    def __init__(self, dim: int, scale: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args:
            t: [B]  continuous time in [0, 1]
        Returns:
            [B, dim]
        """
        device = t.device
        H = self.dim // 2
        alpha = math.log(10000) / (H - 1)
        freq = torch.exp(torch.arange(H, device=device) * -alpha)
        emb = (t * self.scale)[:, None] * freq[None, :]   # [B, H]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)   # [B, dim]


# ─────────────────────────────────────────────────────────────────────────────
# Image encoder
# ─────────────────────────────────────────────────────────────────────────────

class ImageConditioningUNet(nn.Module):
    """DeepLabV3-ResNet101 backbone used as a dense feature extractor.

    Output: [B, 512, H/8, W/8] feature map for cross-attention conditioning.
    """

    def __init__(self, out_channels: int = 512):
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_resnet101
        model = deeplabv3_resnet101(weights="DEFAULT")
        self.backbone = model.backbone  # ResNet-101 feature extractor
        self.proj = nn.Conv2d(2048, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] — normalised to [0, 1]
        feats = self.backbone(x)["out"]  # [B, 2048, H/8, W/8]
        return self.proj(feats)          # [B, out_channels, H/8, W/8]


# ─────────────────────────────────────────────────────────────────────────────
# Transformer block
# ─────────────────────────────────────────────────────────────────────────────

def make_2d_sinusoidal_pos_enc(
    H: int, W: int, dim: int, device: torch.device
) -> torch.Tensor:
    """2D sinusoidal positional encoding for a H×W spatial grid.

    Returns [H*W, dim] tensor whose (row*W+col)-th row encodes the
    normalised coordinates (col/W, row/H) using sinusoidal frequencies.
    This is added to image tokens before cross-attention so the model can
    learn spatial correspondence between anchor positions and image pixels.
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D pos enc"
    half = dim // 2
    freq = torch.exp(
        torch.arange(0, half, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / (half - 2))
    )  # [half//2]
    ys = torch.linspace(-1.0, 1.0, H, device=device)  # normalised [-1,1]
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W] each
    y_flat = grid_y.reshape(-1)  # [H*W]
    x_flat = grid_x.reshape(-1)
    # Build sin/cos for x and y independently
    enc_x = torch.stack([
        torch.sin(x_flat[:, None] * freq[None, :]),
        torch.cos(x_flat[:, None] * freq[None, :]),
    ], dim=-1).reshape(H * W, -1)   # [H*W, half]
    enc_y = torch.stack([
        torch.sin(y_flat[:, None] * freq[None, :]),
        torch.cos(y_flat[:, None] * freq[None, :]),
    ], dim=-1).reshape(H * W, -1)   # [H*W, half]
    return torch.cat([enc_x, enc_y], dim=-1)  # [H*W, dim]


class CrossAttentionBlock(nn.Module):
    """Self-attention on segment tokens + cross-attention to image features."""

    def __init__(self, vector_dim: int = 512, img_dim: int = 512):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(vector_dim, num_heads=8, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(vector_dim, num_heads=8, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(vector_dim, vector_dim * 4),
            nn.GELU(),
            nn.Linear(vector_dim * 4, vector_dim),
        )
        self.norm1 = nn.LayerNorm(vector_dim)
        self.norm2 = nn.LayerNorm(vector_dim)
        self.norm3 = nn.LayerNorm(vector_dim)
        self.img_proj = nn.Linear(img_dim, vector_dim) if img_dim > 0 else None

    def forward(self, x: torch.Tensor, img_features: torch.Tensor | None) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(x, x, x)[0]
        x = self.norm1(x)

        # Cross-attention with image (if provided)
        if img_features is not None and self.img_proj is not None:
            if img_features.dim() == 4:
                # [B, C, H, W] → pool → [B, N, C]
                pooled = F.avg_pool2d(img_features, kernel_size=4)
                img_flat = pooled.flatten(2).permute(0, 2, 1)
            else:
                img_flat = img_features  # already [B, N, C]
            kv = self.img_proj(img_flat)
            x = x + self.cross_attn(x, kv, kv)[0]
        x = self.norm2(x)

        # MLP
        x = x + self.mlp(x)
        x = self.norm3(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full transformer
# ─────────────────────────────────────────────────────────────────────────────

class TransformerModel(nn.Module):
    """
    Transformer denoiser for vector road-segment diffusion.

    Inputs:
        x:             [B, N, 5]  noisy segments (x1, y1, x2, y2, layer)
        t:             [B]        diffusion timestep
        image:         [B, 3, H, W] satellite image (values in [0, 1])

    Outputs:
        eps_pred:      [B, N, 5]  predicted noise
        mask_logits:   [B, N]     per-segment presence logit (1=road, 0=padding)
    """

    def __init__(self, max_segments: int = 500, img_feature_dim: int = 512):
        super().__init__()
        self.max_segments = max_segments

        # Segment input → hidden dim
        self.segment_embed = nn.Linear(5, 512)

        # Positional & time embeddings
        self.pos_embedding = nn.Embedding(max_segments, 512)
        self.time_embedding = TimestepEmbedding(128)   # continuous t ∈ [0,1]
        self.time_embed = nn.Sequential(
            nn.Linear(128, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
        )

        # Image encoder
        self.image_encoder = ImageConditioningUNet(out_channels=img_feature_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            CrossAttentionBlock(512, img_feature_dim) for _ in range(8)
        ])

        # Dedicated spatial coordinate projection so cross-attention queries
        # can explicitly align with the 2D sinusoidal pos_enc in image keys.
        # Uses default (Kaiming) init so it immediately contributes a real
        # spatial signal to cross-attention — essential for the model to attend
        # to the image region near each segment's current position.
        self.coord_embed = nn.Linear(4, 512)

        # Single velocity output head — predicts v for all 5 channels
        self.output_proj = nn.Linear(512, 5)

    def forward(
        self,
        x: torch.Tensor,                    # [B, N, 5]  x_t
        t: torch.Tensor,                    # [B]        t ∈ [0, 1]
        image: torch.Tensor | None = None,  # [B, 3, H, W]  normalised [0,1]
    ) -> torch.Tensor:
        """Predict flow velocity v_θ(x_t, t, image).

        Returns:
            v_pred: [B, N, 5]
        """
        B, N, _ = x.shape

        # Segment + positional embedding
        x_emb = self.segment_embed(x)       # [B, N, 512]
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        x_emb = x_emb + self.pos_embedding(positions)

        # Timestep embedding — t is continuous in [0, 1]
        t_emb = self.time_embed(self.time_embedding(t))   # [B, 512]
        x_emb = x_emb + t_emb.unsqueeze(1)

        # Inject spatial coordinates as a dedicated signal so cross-attention
        # queries can align with the 2D pos_enc in image keys.
        # x[:, :, :4] = (x1, y1, x2, y2) in normalised [-1, 1] space.
        x_emb = x_emb + self.coord_embed(x[:, :, :4])   # [B, N, 512]

        # Encode image
        img_flat = None
        if image is not None:
            img_feats = self.image_encoder(image)              # [B, C, H/8, W/8]
            B_, C, H, W = img_feats.shape
            img_flat = img_feats.view(B_, C, H * W).transpose(1, 2)  # [B, H*W, C]
            # Add 2D sinusoidal positional encoding so cross-attention can learn
            # spatial correspondence between anchor coordinates and image pixels.
            pos_enc = make_2d_sinusoidal_pos_enc(H, W, C, img_flat.device)  # [H*W, C]
            img_flat = img_flat + pos_enc.unsqueeze(0)                       # [B, H*W, C]

        # Transformer with gradient checkpointing during training
        for layer in self.layers:
            if self.training:
                x_emb = torch.utils.checkpoint.checkpoint(
                    layer, x_emb, img_flat, use_reentrant=False
                )
            else:
                x_emb = layer(x_emb, img_flat)

        return self.output_proj(x_emb)   # [B, N, 5]  velocity prediction
