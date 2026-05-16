# src/models/model.py
# Horizon Forecast — Cascaded Dual-Supervision Architecture (Phase C)
# Authors: Or Mordechay Hod, Gilad Boudman  |  Braude College, CODE: 26-1-R-1
#
# Architecture: SimVPv2 Encoder (gSTA) → Stage 1 Physics Head → Stage 2 Manifestation Head
# Input:  (B, 12, 256, 256)  — 4 stacked frames × 3 channels (IR, WV, DEM)
# Output: wind+temp (B,2,256,256) | cloud (B,1,256,256) | rain logits (B,64,256,256)

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

# Gradient checkpointing toggle. Set False to disable (single-step, smaller rollout).
# When True, gSTA blocks + decoders recompute forward during backward, saving ~70%
# activation memory at cost of ~30% training throughput. Required to fit rollout >= 4
# on 24 GB A5000 with bf16. State-dict compatible with non-checkpointed weights —
# pure forward-pass behavior change, no parameter rename.
USE_GRAD_CKPT = True


# ══════════════════════════════════════════════════════════════════════════════
# 1. gSTA Block — Gated Spatiotemporal Attention (SimVPv2 core module)
# ══════════════════════════════════════════════════════════════════════════════
class gSTABlock(nn.Module):
    """
    Gated Spatiotemporal Attention block from SimVPv2 (§2.2, §6.2 of project doc).

    Approximates self-attention via multi-scale depthwise convolutions + gating.
    Complexity: O(N) vs O(N²) for full self-attention — critical for 256×256 inputs.

    Three components per the paper:
      1. Depth-wise conv k=3       → local receptive field (cloud morphology)
      2. Depth-wise dilated conv   → distant connections (steering currents)
      3. Sigmoid gate              → selective feature filtering
      4. 1×1 projection           → channel mixing

    Input/output shape: (B, C, H, W)
    """

    def __init__(self, dim: int, expand_ratio: int = 4, dilation: int = 3):
        super().__init__()
        hidden = dim * expand_ratio

        # Single projection for both feature + gate paths (efficiency)
        self.proj_in    = nn.Conv2d(dim, hidden * 2, 1, bias=False)

        # Multi-scale depthwise: local + distant
        self.dw_local   = nn.Conv2d(hidden, hidden, 3, padding=1,
                                    groups=hidden, bias=False)
        self.dw_distant = nn.Conv2d(hidden, hidden, 3, padding=dilation,
                                    dilation=dilation, groups=hidden, bias=False)

        self.proj_out   = nn.Conv2d(hidden, dim, 1, bias=False)

        # GroupNorm instead of BatchNorm: stable at any batch size (important for B=1 inference)
        self.norm1 = nn.GroupNorm(min(8, dim), dim)
        self.norm2 = nn.GroupNorm(min(8, dim), dim)

        # Position-wise FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x   = self.norm1(x)

        feat, gate = self.proj_in(x).chunk(2, dim=1)   # each (B, hidden, H, W)
        feat = self.dw_local(feat) + self.dw_distant(feat)
        feat = feat * torch.sigmoid(gate)               # gated activation
        x    = self.proj_out(feat) + res                # residual

        x = x + self.ffn(self.norm2(x))                # FFN residual
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 2. SimVPv2 Encoder — Visual Body
# ══════════════════════════════════════════════════════════════════════════════
class SimVPv2Encoder(nn.Module):
    """
    Spatiotemporal feature extractor (§6.2 of project doc).

    Processes the 4D stacked-image tensor. The temporal dimension is encoded
    in channels (C_stacked=12), not as a separate video dimension — this is
    a strict architectural choice for Phase B/C.

    Pipeline:
      Stem: 256×256 → 64×64  (2× stride-2 conv, 4× spatial compression)
      Body: 8 gSTA blocks    (capture multi-scale atmospheric patterns)
      Output Z: (B, 256, 64, 64) — feeds both Stage 1 and Stage 2

    The 4× downsampling keeps memory tractable while preserving enough
    spatial resolution for the heads to reconstruct accurate 256×256 maps.
    """

    def __init__(
        self,
        in_channels: int = 12,   # C_STACKED = T_IN * (C_SAT + C_STATIC)
        latent_dim:  int = 256,
        n_blocks:    int = 8,
        dilation:    int = 3,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),        # 256→128
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, latent_dim, 3, stride=2, padding=1, bias=False), # 128→64
            nn.GroupNorm(8, latent_dim),
            nn.GELU(),
        )

        # ModuleList (not Sequential) so forward can iterate + grad-checkpoint each
        # block individually. State-dict keys identical to Sequential (numeric).
        self.blocks = nn.ModuleList(
            [gSTABlock(latent_dim, dilation=dilation) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 12, 256, 256) → Z: (B, 256, 64, 64)"""
        z = self.stem(x)
        if USE_GRAD_CKPT and self.training:
            for blk in self.blocks:
                z = cp.checkpoint(blk, z, use_reentrant=False)
        else:
            for blk in self.blocks:
                z = blk(z)
        return z


# ══════════════════════════════════════════════════════════════════════════════
# 3. Stage 1 — Physics Driver Head + Masked MSE Loss
# ══════════════════════════════════════════════════════════════════════════════
class PhysicsDriverHead(nn.Module):
    """
    Stage 1 cascade: predicts thermodynamic atmospheric drivers (§6.3).

    Input:  Z (B, 256, 64, 64)  — latent features from encoder
    Output: (B, 2, 256, 256)
              channel 0: Surface Wind Speed (m/s, normalized)
              channel 1: Surface Temperature (°C, normalized)

    Sparse Supervision Strategy:
      Ground truth exists only at ~60-80 IMS station pixels per 256×256 frame.
      MaskedMSELoss ensures gradients flow ONLY from those pixels.
      Convolutional diffusion naturally interpolates between stations —
      the CNN fills the map via learned spatial priors.

    Architecture is deliberately shallow (4 gSTA + decoder) to avoid
    overfitting sparse IMS supervision.
    """

    def __init__(self, latent_dim: int = 256, n_blocks: int = 4):
        super().__init__()

        self.refine = nn.ModuleList(
            [gSTABlock(latent_dim) for _ in range(n_blocks)]
        )

        # Bilinear upsample + 3×3 conv: 64×64 → 256×256.
        # Replaces ConvTranspose2d (kernel=4, stride=2) which produces
        # checkerboard artifacts due to uneven kernel/stride overlap.
        # Smooth interpolation + learned refinement = no aliasing.
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(latent_dim, 128, 3, padding=1, bias=False),                      # →128
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),                              # →256
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),  # extra smoothing conv
            nn.GELU(),
            nn.Conv2d(64, 2, 1),  # wind + temp — no activation (regression output)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Z: (B, 256, 64, 64) → (B, 2, 256, 256)"""
        if USE_GRAD_CKPT and self.training:
            for blk in self.refine:
                z = cp.checkpoint(blk, z, use_reentrant=False)
            return cp.checkpoint(self.decoder, z, use_reentrant=False)
        for blk in self.refine:
            z = blk(z)
        return self.decoder(z)


class MaskedMSELoss(nn.Module):
    """
    Sparse supervision loss for Stage 1 (§6.6 eq. 4).

    Computes MSE only at pixels where station_mask=True.
    Zeros in the target outside station locations are ignored —
    gradients never propagate from empty grid regions.

    pred:   (B, 2, H, W)  model output
    target: (B, 2, H, W)  ground truth (non-zero only at station pixels)
    mask:   (B, H, W) bool  True at active IMS station locations
    """

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        mask:   torch.Tensor,
    ) -> torch.Tensor:
        m = mask.unsqueeze(1).float()            # (B, 1, H, W) — broadcast over channels
        n = (m.sum() * pred.size(1)).clamp(min=1.0)
        return ((pred - target).pow(2) * m).sum() / n


# ══════════════════════════════════════════════════════════════════════════════
# 4. Stage 2 — Manifestation Head (SaTformer-Inspired)
# ══════════════════════════════════════════════════════════════════════════════
class _SpatialTransformerBlock(nn.Module):
    """
    Transformer block for global storm cluster detection (§2.3, §6.5).

    SaTformer insight: precipitation forecasting requires GLOBAL context —
    storm clusters form and dissipate across long distances. CNN receptive
    fields are insufficient. Transformer patch attention solves this.

    Operates on 8×8=64 patch tokens extracted from the 64×64 latent feature
    map (patch_size=8). Full 64²=4096 token attention is skipped — 64 tokens
    capture global context at affordable cost.
    """

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True,
                                           dropout=0.1, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(dim * 4, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, dim)"""
        n = self.norm1(x)
        x = x + self.attn(n, n, n)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ManifestationHead(nn.Module):
    """
    Stage 2 cascade: fused context → cloud structure + rain intensity (§6.4, §6.5).

    Cascade Fusion (§6.4 eq. 1):
      Input_manifestation = Z ⊕ Ŷ_wind ⊕ Ŷ_temp

      Drivers are projected from 256×256 back to 64×64 latent space,
      concatenated with Z, then blended via 1×1 conv. This forces the
      model to see WHERE the wind is pushing before predicting WHERE rain falls.

    Two-branch decoder:
      cloud_head : regression  → (B, 2, 256, 256)  IR+WV — both channels predicted so the
                                  model can feed its own output back as input for autoregressive
                                  rollout (scheduled sampling training, Option A)
      rain_head  : 64 logits   → (B, 64, 256, 256) for sparse CE at station pixels

    rain_head outputs 64 probability logit maps — one per intensity bucket.
    Argmax at inference gives predicted rain class per pixel.
    """

    def __init__(
        self,
        latent_dim:  int = 256,
        n_rain_bins: int = 64,
        n_heads:     int = 8,
        n_xfmr:      int = 4,
        patch_size:  int = 8,   # 64/8 = 8×8 = 64 patch tokens
        n_gsta:      int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Project drivers (256×256) → latent resolution (64×64)
        self.driver_proj = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1, bias=False),          # 256→128
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, latent_dim, 3, stride=2, padding=1, bias=False),  # 128→64
            nn.GroupNorm(8, latent_dim),
        )

        # Cascade fusion: [Z; proj_drivers] → single latent
        self.fusion = nn.Sequential(
            nn.Conv2d(latent_dim * 2, latent_dim, 1, bias=False),
            nn.GroupNorm(8, latent_dim),
            nn.GELU(),
        )

        # Local spatial refinement via gSTA
        self.spatial_refine = nn.ModuleList(
            [gSTABlock(latent_dim) for _ in range(n_gsta)]
        )

        # Patch embedding: 64×64 → 8×8 non-overlapping patches = 64 tokens
        self.patch_embed = nn.Conv2d(
            latent_dim, latent_dim, patch_size, stride=patch_size, bias=False
        )

        # Global transformer for storm cluster context
        self.xfmr = nn.ModuleList(
            [_SpatialTransformerBlock(latent_dim, n_heads) for _ in range(n_xfmr)]
        )

        # Bilinear upsample + 3×3 conv: 64×64 → 256×256.
        # Replaces ConvTranspose2d to eliminate checkerboard artifacts.
        # Extra final conv smooths any residual high-frequency aliasing
        # before cloud + rain heads project to output channels.
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(latent_dim, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),  # extra smoothing conv
            nn.GELU(),
        )

        self.cloud_head = nn.Conv2d(64, 2, 1)           # cloud regression: IR+WV (2ch)
        self.rain_head  = nn.Conv2d(64, n_rain_bins, 1) # rain 64-class logits

        # Sinusoidal positional embedding for patch tokens.
        # patch_embed always outputs 8×8 for fixed 64×64 input → n=64 is constant.
        # Pre-computed as buffer so forward() is mutation-free (required by torch.compile).
        dim = self.patch_embed.out_channels
        n   = (64 // patch_size) ** 2
        pos = torch.arange(n).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe  = torch.zeros(1, n, dim)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[: dim // 2])
        self.register_buffer("pos_embed", pe)  # (1, 64, 256)

    def forward(
        self, z: torch.Tensor, drivers: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = z.size(0)

        # Cascade fusion: project drivers → concat → blend
        d     = self.driver_proj(drivers)                        # (B, 256, 64, 64)
        fused = self.fusion(torch.cat([z, d], dim=1))            # (B, 256, 64, 64)
        if USE_GRAD_CKPT and self.training:
            for blk in self.spatial_refine:
                fused = cp.checkpoint(blk, fused, use_reentrant=False)
        else:
            for blk in self.spatial_refine:
                fused = blk(fused)

        # Patch attention: global storm cluster context
        p      = self.patch_embed(fused)                         # (B, 256, 8, 8)
        pH, pW = p.shape[2], p.shape[3]
        tokens = p.flatten(2).permute(0, 2, 1)                  # (B, 64, 256)
        tokens = tokens + self.pos_embed

        for blk in self.xfmr:
            tokens = blk(tokens)

        # Reshape + upsample back to 64×64, add residual
        p_out = tokens.permute(0, 2, 1).reshape(B, -1, pH, pW)
        p_up  = F.interpolate(p_out, size=(64, 64), mode="bilinear", align_corners=False)
        fused = fused + p_up

        # Decode to full 256×256 resolution (checkpointed — heaviest activation)
        if USE_GRAD_CKPT and self.training:
            feat = cp.checkpoint(self.upsample, fused, use_reentrant=False)
        else:
            feat = self.upsample(fused)         # (B, 64, 256, 256)
        y_cloud = self.cloud_head(feat)         # (B, 1, 256, 256)
        y_rain  = self.rain_head(feat)          # (B, 64, 256, 256)
        return y_cloud, y_rain


# ══════════════════════════════════════════════════════════════════════════════
# 5. Full Horizon Forecast Model
# ══════════════════════════════════════════════════════════════════════════════
class HorizonForecastModel(nn.Module):
    """
    Horizon Forecast: Cascaded Dual-Supervision Network.
    CODE: 26-1-R-1  |  Braude College of Engineering, Software Engineering Dept.

    Full forward pass:
      x (B, 12, 256, 256)
        → SimVPv2Encoder    → Z (B, 256, 64, 64)
        → PhysicsDriverHead → drivers (B, 2, 256, 256)   [wind | temp]
        → ManifestationHead(Z, drivers)
            → y_cloud  (B, 2, 256, 256)   [IR+WV for next timestamp — both channels
                                            so output feeds directly back as input for
                                            autoregressive scheduled-sampling rollout]
            → y_rain   (B, 64, 256, 256)  [rain intensity class logits]

    The Driver-First Hypothesis (§1.3, §6.1):
      Standard models predict optical flow (pixels → pixels), ignoring physics.
      This model first predicts WHY the storm moves (wind/temperature), then
      predicts WHERE it goes (clouds/rain). Physical causality is enforced by
      architecture, not just loss functions.

    Parameter count: ~30.1M params at default settings (in_channels=12,
    latent_dim=256, n_rain_bins=64). Verified at runtime via model.n_params.
    VRAM at BF16 B=16: ~60 MB weights + ~22 GB activations on A5000 (92% util).
    VRAM at FP16 B=1 inference: ~60 MB weights + ~0.5 GB activations → 3070-safe.
    """

    def __init__(
        self,
        in_channels: int = 12,
        latent_dim:  int = 256,
        n_rain_bins: int = 64,
        no_cascade:  bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim  = latent_dim
        self.n_rain_bins = n_rain_bins
        # Ablation flag: when True, zero out driver tensor fed to ManifestationHead.
        # Stage 1 still supervised by L_thermo, but Stage 2 cannot see driver predictions.
        # Isolates the cascade contribution for the research-track ablation experiment.
        self.no_cascade  = no_cascade
        self.encoder   = SimVPv2Encoder(in_channels, latent_dim)
        self.phys_head = PhysicsDriverHead(latent_dim)
        self.mani_head = ManifestationHead(latent_dim, n_rain_bins)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z                = self.encoder(x)
        drivers          = self.phys_head(z)
        drivers_for_mani = torch.zeros_like(drivers) if self.no_cascade else drivers
        y_cloud, y_rain  = self.mani_head(z, drivers_for_mani)
        return drivers, y_cloud, y_rain

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
