"""
Mamba block for temporal sequence processing.

Based on: https://github.com/state-spaces/mamba
Adapted from: https://github.com/IRMVLab/Mamba4D

The Mamba block uses Selective State Space Models (SSM) for efficient
sequence modeling with O(n) complexity instead of O(n^2) for transformers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from functools import partial

# Try to import mamba-ssm
try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# Try to import timm for DropPath
try:
    from timm.models.layers import DropPath
except ImportError:
    # Fallback DropPath implementation
    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
            return output


class Block(nn.Module):
    """
    Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection.

    Structure: Add -> LN -> Mixer
    This order allows fusing add and LayerNorm for performance.
    """

    def __init__(
        self,
        dim: int,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        drop_path: float = 0.
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), \
                "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inference_params=None
    ):
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def _init_weights(
    module,
    n_layer,
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,
):
    """Initialize weights for Mamba blocks."""
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
    d_model: int,
    ssm_cfg: dict = None,
    norm_epsilon: float = 1e-5,
    rms_norm: bool = False,
    residual_in_fp32: bool = False,
    fused_add_norm: bool = False,
    layer_idx: int = None,
    drop_path: float = 0.,
    device=None,
    dtype=None,
):
    """Create a single Mamba block."""
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    if MAMBA_AVAILABLE:
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    else:
        # Use fallback
        mixer_cls = partial(MambaFallback, **ssm_cfg)

    if rms_norm and RMSNorm is not None:
        norm_cls = partial(RMSNorm, eps=norm_epsilon, **factory_kwargs)
    else:
        norm_cls = partial(nn.LayerNorm, eps=norm_epsilon)

    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm and MAMBA_AVAILABLE,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    """
    Stacked Mamba mixer model.

    This is the core temporal encoder that processes sequences using
    selective state space models.
    """

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        ssm_cfg: dict = None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg: dict = None,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        drop_out_in_block: float = 0.,
        drop_path: float = 0.1,
        drop_path_rate: float = 0.1,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm and MAMBA_AVAILABLE

        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]

        self.layers = nn.ModuleList([
            create_block(
                d_model,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=self.fused_add_norm,
                layer_idx=i,
                drop_path=dpr[i],
                **factory_kwargs,
            )
            for i in range(n_layer)
        ])

        if rms_norm and RMSNorm is not None:
            self.norm_f = RMSNorm(d_model, eps=norm_epsilon, **factory_kwargs)
        else:
            self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, pos: torch.Tensor = None, inference_params=None):
        """
        Args:
            x: (B, T, D) input sequence
            pos: (B, T, D) optional positional encoding
            inference_params: Optional inference parameters for caching

        Returns:
            (B, T, D) output sequence
        """
        hidden_states = x
        residual = None

        if pos is not None:
            hidden_states = hidden_states + pos

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


class MambaBlock(nn.Module):
    """
    Single Mamba block for sequence modeling with residual connection.

    A simpler wrapper that handles the forward pass with built-in normalization
    and residual connections.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: Input/output dimension
            d_state: SSM state dimension
            d_conv: Depth-wise convolution kernel size
            expand: Expansion factor for inner dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.dim = dim
        self.d_inner = dim * expand

        self.norm = nn.LayerNorm(dim)

        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba = MambaFallback(dim, d_state, d_conv, expand)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input sequence

        Returns:
            y: (B, T, D) output sequence
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaFallback(nn.Module):
    """
    Fallback implementation when mamba-ssm is not available.

    Uses GRU as approximation (not truly selective SSM, but similar behavior).
    This allows the code to run without CUDA-compiled Mamba kernels.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs  # Accept extra kwargs for compatibility
    ):
        super().__init__()
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.gru = nn.GRU(d_inner, d_inner, batch_first=True)
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input sequence

        Returns:
            (B, T, D) output sequence
        """
        B, T, D = x.shape

        # Input projection with gating
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # Depthwise conv
        x_conv = x_proj.transpose(1, 2)  # (B, D, T)
        x_conv = self.conv(x_conv)[:, :, :T]  # Causal padding
        x_conv = x_conv.transpose(1, 2)  # (B, T, D)

        # GRU (approximates SSM)
        x_conv = F.silu(x_conv)
        x_out, _ = self.gru(x_conv)

        # Gate and project
        x_out = x_out * F.silu(z)
        return self.out_proj(x_out)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Placeholder for compatibility with Mamba interface."""
        return None
