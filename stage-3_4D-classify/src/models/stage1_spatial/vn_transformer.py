"""
VN-Transformer wrapper adapted for 3D Gaussian classification.

This module wraps the VN-Transformer to process per-frame Gaussian features
and produce rotation-invariant frame embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import rearrange

from .vn_layers import (
    VNTransformer,
    VNLinear,
    VNReLU,
    VNLayerNorm,
    VNWeightedPool,
    VNInvariant
)


class GaussianSpatialEncoder(nn.Module):
    """
    Stage 1: VN-Transformer-based spatial encoder for Gaussians.

    Takes per-frame Gaussian features and produces rotation-invariant frame embeddings.

    Architecture:
        1. Project scalar features to VN-compatible dimension
        2. Process through VN-Transformer (equivariant processing)
        3. Attention pooling to aggregate per-Gaussian features to frame-level
        4. VN-Invariant projection to produce rotation-invariant output
    """

    def __init__(
        self,
        dim: int = 128,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 8,
        dim_feat: int = 64,
        bias_epsilon: float = 1e-6,
        dropout: float = 0.1,
        output_dim: int = 256,
        l2_dist_attn: bool = False,
        flash_attn: bool = False,
    ):
        """
        Args:
            dim: VN-Transformer hidden dimension (per vector channel)
            depth: Number of VN-Transformer layers
            dim_head: Dimension per attention head
            heads: Number of attention heads
            dim_feat: Dimension for scalar (non-spatial) features
            bias_epsilon: Small bias for numerical stability (ε-approximate equivariance)
            dropout: Dropout rate
            output_dim: Dimension of output frame embedding
            l2_dist_attn: Use L2 distance-based attention
            flash_attn: Use flash attention (requires PyTorch 2.0+)
        """
        super().__init__()

        self.dim = dim
        self.dim_feat = dim_feat
        self.output_dim = output_dim

        # Feature preparation: project scalar features to VN-compatible dimension
        # Input: eigenvals(3) + opacity(1) + sh_bands(12) = 16
        self.scalar_proj = nn.Linear(16, dim_feat)

        # Main VN-Transformer
        self.vn_transformer = VNTransformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            dim_feat=dim_feat,
            bias_epsilon=bias_epsilon,
            l2_dist_attn=l2_dist_attn,
            flash_attn=flash_attn,
            reduce_dim_out=True,
        )

        # Frame-level aggregation (attention pooling)
        self.attention_pool = VNAttentionPooling(
            dim=dim,
            output_dim=output_dim,
            dim_feat=dim_feat,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        positions: torch.Tensor,
        scalars: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process a single frame's Gaussians.

        Args:
            positions: (B, M, 3) Gaussian positions
            scalars: (B, M, D_scalar) Scalar features
            mask: (B, M) Boolean mask for valid Gaussians

        Returns:
            frame_embedding: (B, output_dim) rotation-invariant frame embedding
        """
        B, M, _ = positions.shape

        # Prepare scalar features
        scalar_feats = self.scalar_proj(scalars)  # (B, M, dim_feat)

        # VN-Transformer expects:
        #   coors: (B, M, 3) - spatial coordinates
        #   feats: (B, M, dim_feat) - scalar features
        coors_out, feats_out = self.vn_transformer(
            positions,
            feats=scalar_feats,
            mask=mask,
        )
        # coors_out: (B, M, 3) - equivariant
        # feats_out: (B, M, dim_feat) - mixed equivariant/invariant

        # Aggregate to frame-level embedding
        frame_embedding = self.attention_pool(coors_out, feats_out, mask)

        return self.dropout(frame_embedding)

    def forward_sequence(
        self,
        positions: torch.Tensor,
        scalars: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process a sequence of frames.

        Args:
            positions: (B, T, M, 3) Gaussian positions per frame
            scalars: (B, T, M, D_scalar) Scalar features per frame
            mask: (B, T, M) Boolean mask for valid Gaussians

        Returns:
            frame_embeddings: (B, T, output_dim) sequence of frame embeddings
        """
        B, T, M, _ = positions.shape

        # Process each frame
        embeddings = []
        for t in range(T):
            pos_t = positions[:, t]  # (B, M, 3)
            scalar_t = scalars[:, t]  # (B, M, D_scalar)
            mask_t = mask[:, t] if mask is not None else None

            emb_t = self.forward(pos_t, scalar_t, mask_t)
            embeddings.append(emb_t)

        # Stack: (B, T, output_dim)
        return torch.stack(embeddings, dim=1)


class VNAttentionPooling(nn.Module):
    """
    Attention-weighted pooling to aggregate per-Gaussian features to frame-level.

    Produces rotation-invariant output through a combination of:
    1. Attention-weighted aggregation of features
    2. Extraction of rotation-invariant geometric properties
    """

    def __init__(
        self,
        dim: int,
        output_dim: int,
        dim_feat: int = 64,
    ):
        super().__init__()

        # Attention weights (computed from features)
        self.attention_mlp = nn.Sequential(
            nn.Linear(dim_feat, dim_feat // 2),
            nn.ReLU(),
            nn.Linear(dim_feat // 2, 1),
        )

        # Final projection to output dimension
        # Input: dim_feat (from features) + 3 (spatial invariants)
        self.output_proj = nn.Sequential(
            nn.Linear(dim_feat + 3, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        coors: torch.Tensor,
        feats: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            coors: (B, M, 3) equivariant coordinates from VN-Transformer
            feats: (B, M, dim_feat) features from VN-Transformer
            mask: (B, M) validity mask

        Returns:
            embedding: (B, output_dim) rotation-invariant embedding
        """
        B, M, _ = coors.shape

        # Compute attention weights from features
        attn_logits = self.attention_mlp(feats).squeeze(-1)  # (B, M)

        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))

        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, M)

        # Handle case where all positions are masked
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Weighted aggregation of coordinates
        coors_agg = torch.einsum('bm,bmd->bd', attn_weights, coors)  # (B, 3)

        # Weighted aggregation of features
        feats_agg = torch.einsum('bm,bmd->bd', attn_weights, feats)  # (B, dim_feat)

        # Compute rotation-invariant spatial statistics
        # 1. Norm of aggregated coordinates (overall spatial extent)
        coors_norm = coors_agg.norm(dim=-1, keepdim=True)  # (B, 1)

        # 2. Weighted variance of coordinates (spatial spread)
        coors_centered = coors - coors_agg.unsqueeze(1)  # (B, M, 3)
        coors_var = torch.einsum('bm,bmd->bd', attn_weights, coors_centered ** 2)  # (B, 3)
        spatial_spread = coors_var.sum(dim=-1, keepdim=True)  # (B, 1)

        # 3. Weighted mean distance from center
        distances = coors_centered.norm(dim=-1)  # (B, M)
        mean_dist = torch.einsum('bm,bm->b', attn_weights, distances).unsqueeze(-1)  # (B, 1)

        # Concatenate invariant features
        spatial_invariants = torch.cat([coors_norm, spatial_spread, mean_dist], dim=-1)  # (B, 3)

        # Combine features and spatial invariants
        combined = torch.cat([feats_agg, spatial_invariants], dim=-1)  # (B, dim_feat + 3)

        # Project to output dimension
        output = self.output_proj(combined)

        return output
