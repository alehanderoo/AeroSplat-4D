"""
Stage 1: VN-Transformer Spatial Encoder

Rotation-invariant spatial encoding of per-frame Gaussian primitives.
"""

from .vn_transformer import GaussianSpatialEncoder, VNAttentionPooling
from .vn_layers import (
    VNLinear,
    VNReLU,
    VNLayerNorm,
    VNAttention,
    VNFeedForward,
    VNWeightedPool,
    VNTransformerEncoder,
    VNInvariant
)
from .feature_preparation import GaussianFeaturePreparation

__all__ = [
    'GaussianSpatialEncoder',
    'VNAttentionPooling',
    'VNLinear',
    'VNReLU',
    'VNLayerNorm',
    'VNAttention',
    'VNFeedForward',
    'VNWeightedPool',
    'VNTransformerEncoder',
    'VNInvariant',
    'GaussianFeaturePreparation',
]
