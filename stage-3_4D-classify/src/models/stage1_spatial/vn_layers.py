"""
Vector Neuron layers for rotation-equivariant processing.

These layers implement the Vector Neuron framework where features are represented
as (N, C, 3) tensors - each channel is a 3D vector that transforms equivariantly
under rotations.

Adapted from: https://github.com/lucidrains/VN-transformer
"""

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from .attend import Attend


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def inner_dot_product(x, y, *, dim=-1, keepdim=True):
    """Compute inner product along specified dimension."""
    return (x * y).sum(dim=dim, keepdim=keepdim)


class LayerNorm(nn.Module):
    """Standard LayerNorm without learnable bias."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class VNLinear(nn.Module):
    """
    Vector Neuron Linear layer.

    Performs channel mixing while preserving the 3D spatial structure.
    Each 3D vector is transformed by the same weight matrix across spatial dimensions.

    Optional: ε-approximate equivariance via small bias for numerical stability.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias_epsilon: float = 0.
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim_in))

        self.bias = None
        self.bias_epsilon = bias_epsilon

        # Small bias for quasi-equivariance (better stability and results per paper)
        if bias_epsilon > 0.:
            self.bias = nn.Parameter(torch.randn(dim_out))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, C_in, 3) input tensor
        Returns:
            (B, N, C_out, 3) output tensor
        """
        out = einsum('... i c, o i -> ... o c', x, self.weight)

        if exists(self.bias):
            bias = F.normalize(self.bias, dim=-1) * self.bias_epsilon
            out = out + rearrange(bias, '... -> ... 1')

        return out


class VNReLU(nn.Module):
    """
    Vector Neuron ReLU - equivariant nonlinearity.

    Uses learned projection directions to determine activation.
    When qk >= 0: output q
    When qk < 0: output projection of q onto hyperplane orthogonal to k
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.W = nn.Parameter(torch.randn(dim, dim))
        self.U = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        q = einsum('... i c, o i -> ... o c', x, self.W)
        k = einsum('... i c, o i -> ... o c', x, self.U)

        qk = inner_dot_product(q, k)

        k_norm = k.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        q_projected_on_k = q - inner_dot_product(q, k / k_norm) * k

        out = torch.where(
            qk >= 0.,
            q,
            q_projected_on_k
        )

        return out


class VNLayerNorm(nn.Module):
    """
    Vector Neuron LayerNorm.

    Normalizes the norm of each 3D vector, then applies standard LayerNorm
    to the norms.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.ln = LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        norms = x.norm(dim=-1)
        x = x / rearrange(norms.clamp(min=self.eps), '... -> ... 1')
        ln_out = self.ln(norms)
        return x * rearrange(ln_out, '... -> ... 1')


class VNAttention(nn.Module):
    """
    Vector Neuron Attention - rotation-invariant attention mechanism.

    Uses Frobenius inner products (matrix inner products) to compute
    rotation-invariant attention weights.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dim_coor: int = 3,
        bias_epsilon: float = 0.,
        l2_dist_attn: bool = False,
        flash: bool = False,
        num_latents: int = None
    ):
        super().__init__()
        assert not (l2_dist_attn and flash), 'l2 distance attention is not compatible with flash attention'

        self.scale = (dim_coor * dim_head) ** -0.5
        dim_inner = dim_head * heads
        self.heads = heads

        self.to_q_input = None
        if exists(num_latents):
            self.to_q_input = VNWeightedPool(dim, num_pooled_tokens=num_latents, squeeze_out_pooled_dim=False)

        self.to_q = VNLinear(dim, dim_inner, bias_epsilon=bias_epsilon)
        self.to_k = VNLinear(dim, dim_inner, bias_epsilon=bias_epsilon)
        self.to_v = VNLinear(dim, dim_inner, bias_epsilon=bias_epsilon)
        self.to_out = VNLinear(dim_inner, dim, bias_epsilon=bias_epsilon)

        if l2_dist_attn and not exists(num_latents):
            # tied queries and keys for l2 distance attention
            self.to_k = self.to_q

        self.attend = Attend(flash=flash, l2_dist=l2_dist_attn)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x: (B, N, C, 3) input tensor
            mask: (B, N) optional mask
        Returns:
            (B, N, C, 3) output tensor
        """
        c = x.shape[-1]

        if exists(self.to_q_input):
            q_input = self.to_q_input(x, mask=mask)
        else:
            q_input = x

        q, k, v = self.to_q(q_input), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) c -> b h n (d c)', h=self.heads), (q, k, v))

        out = self.attend(q, k, v, mask=mask)

        out = rearrange(out, 'b h n (d c) -> b n (h d) c', c=c)
        return self.to_out(out)


def VNFeedForward(dim: int, mult: int = 4, bias_epsilon: float = 0.):
    """Vector Neuron FeedForward block."""
    dim_inner = int(dim * mult)
    return nn.Sequential(
        VNLinear(dim, dim_inner, bias_epsilon=bias_epsilon),
        VNReLU(dim_inner),
        VNLinear(dim_inner, dim, bias_epsilon=bias_epsilon)
    )


class VNWeightedPool(nn.Module):
    """
    Vector Neuron Weighted Pooling.

    Aggregates sequence of vectors using learned weights.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int = None,
        num_pooled_tokens: int = 1,
        squeeze_out_pooled_dim: bool = True
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.weight = nn.Parameter(torch.randn(num_pooled_tokens, dim, dim_out))
        self.squeeze_out_pooled_dim = num_pooled_tokens == 1 and squeeze_out_pooled_dim

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            x = x.masked_fill(~mask, 0.)
            numer = reduce(x, 'b n d c -> b d c', 'sum')
            denom = mask.sum(dim=1)
            mean_pooled = numer / denom.clamp(min=1e-6)
        else:
            mean_pooled = reduce(x, 'b n d c -> b d c', 'mean')

        out = einsum('b d c, m d e -> b m e c', mean_pooled, self.weight)

        if not self.squeeze_out_pooled_dim:
            return out

        out = rearrange(out, 'b 1 d c -> b d c')
        return out


class VNTransformerEncoder(nn.Module):
    """
    Vector Neuron Transformer Encoder.

    Stacks VN-Attention and VN-FeedForward blocks with LayerNorm.
    """

    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        dim_coor: int = 3,
        ff_mult: int = 4,
        final_norm: bool = False,
        bias_epsilon: float = 0.,
        l2_dist_attn: bool = False,
        flash_attn: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.dim_coor = dim_coor

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                VNAttention(
                    dim=dim, dim_head=dim_head, heads=heads,
                    bias_epsilon=bias_epsilon, l2_dist_attn=l2_dist_attn,
                    flash=flash_attn
                ),
                VNLayerNorm(dim),
                VNFeedForward(dim=dim, mult=ff_mult, bias_epsilon=bias_epsilon),
                VNLayerNorm(dim)
            ]))

        self.norm = VNLayerNorm(dim) if final_norm else nn.Identity()

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x: (B, N, D, C) tensor where D=dim and C=dim_coor
            mask: (B, N) optional mask
        """
        *_, d, c = x.shape

        assert x.ndim == 4 and d == self.dim and c == self.dim_coor, \
            f'input needs to be in shape (batch, seq, dim ({self.dim}), coordinate dim ({self.dim_coor}))'

        for attn, attn_post_ln, ff, ff_post_ln in self.layers:
            x = attn_post_ln(attn(x, mask=mask)) + x
            x = ff_post_ln(ff(x)) + x

        return self.norm(x)


class VNInvariant(nn.Module):
    """
    Vector Neuron Invariant layer.

    Produces rotation-invariant features from equivariant representations
    by learning a local coordinate frame and projecting onto it.
    """

    def __init__(
        self,
        dim: int,
        dim_coor: int = 3,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            VNLinear(dim, dim_coor),
            VNReLU(dim_coor),
            Rearrange('... d e -> ... e d')
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, D, 3) equivariant features
        Returns:
            (B, N, D) invariant features
        """
        return einsum('b n d i, b n i o -> b n o', x, self.mlp(x))


class VNTransformer(nn.Module):
    """
    Complete VN-Transformer for point cloud processing.

    Combines:
    - Input projection to VN feature space
    - VN-Transformer encoder
    - Output projection

    Supports optional translation equivariance/invariance.
    """

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        num_tokens: int = None,
        dim_feat: int = None,
        dim_head: int = 64,
        heads: int = 8,
        dim_coor: int = 3,
        reduce_dim_out: bool = True,
        bias_epsilon: float = 0.,
        l2_dist_attn: bool = False,
        flash_attn: bool = False,
        translation_equivariance: bool = False,
        translation_invariant: bool = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None

        dim_feat = default(dim_feat, 0)
        self.dim_feat = dim_feat
        self.dim_coor_total = dim_coor + dim_feat

        assert (int(translation_equivariance) + int(translation_invariant)) <= 1
        self.translation_equivariance = translation_equivariance
        self.translation_invariant = translation_invariant

        self.vn_proj_in = nn.Sequential(
            Rearrange('... c -> ... 1 c'),
            VNLinear(1, dim, bias_epsilon=bias_epsilon)
        )

        self.encoder = VNTransformerEncoder(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            bias_epsilon=bias_epsilon,
            dim_coor=self.dim_coor_total,
            l2_dist_attn=l2_dist_attn,
            flash_attn=flash_attn
        )

        if reduce_dim_out:
            self.vn_proj_out = nn.Sequential(
                VNLayerNorm(dim),
                VNLinear(dim, 1, bias_epsilon=bias_epsilon),
                Rearrange('... 1 c -> ... c')
            )
        else:
            self.vn_proj_out = nn.Identity()

    def forward(
        self,
        coors: Tensor,
        *,
        feats: Tensor = None,
        mask: Tensor = None,
        return_concatted_coors_and_feats: bool = False
    ):
        """
        Args:
            coors: (B, N, 3) spatial coordinates
            feats: (B, N, D_feat) optional scalar features
            mask: (B, N) optional mask
            return_concatted_coors_and_feats: if True, return concatenated output

        Returns:
            coors_out: (B, N, 3) transformed coordinates (equivariant)
            feats_out: (B, N, D_feat) transformed features (optional)
        """
        if self.translation_equivariance or self.translation_invariant:
            coors_mean = reduce(coors, '... c -> c', 'mean')
            coors = coors - coors_mean

        x = coors

        if exists(feats):
            if feats.dtype == torch.long:
                assert exists(self.token_emb), 'num_tokens must be given for embedding indices'
                feats = self.token_emb(feats)

            assert feats.shape[-1] == self.dim_feat, f'dim_feat should be {feats.shape[-1]}'
            x = torch.cat((x, feats), dim=-1)

        assert x.shape[-1] == self.dim_coor_total

        x = self.vn_proj_in(x)
        x = self.encoder(x, mask=mask)
        x = self.vn_proj_out(x)

        coors_out, feats_out = x[..., :3], x[..., 3:]

        if self.translation_equivariance:
            coors_out = coors_out + coors_mean

        if not exists(feats):
            return coors_out

        if return_concatted_coors_and_feats:
            return torch.cat((coors_out, feats_out), dim=-1)

        return coors_out, feats_out
