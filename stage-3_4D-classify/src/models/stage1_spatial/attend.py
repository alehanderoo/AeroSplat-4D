"""
Attention module for VN-Transformer.

Adapted from: https://github.com/lucidrains/VN-transformer
"""

from functools import wraps
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

# constants
FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def exists(val):
    return val is not None


def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner


print_once = once(print)


class Attend(nn.Module):
    """
    Attention mechanism supporting both standard and flash attention.
    Also supports L2 distance-based attention for geometric data.
    """

    def __init__(
        self,
        dropout: float = 0.,
        flash: bool = False,
        l2_dist: bool = False
    ):
        super().__init__()
        assert not (flash and l2_dist), 'flash attention is not compatible with l2 distance'
        self.l2_dist = l2_dist

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash

        # determine efficient attention configs for cuda and cpu
        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Check if mask exists and expand to compatible shape
        if exists(mask):
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v, mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')

        if self.flash:
            return self.flash_attn(q, k, v, mask=mask)

        # similarity
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # l2 distance
        if self.l2_dist:
            # -cdist squared == (-q^2 + 2qk - k^2)
            q_squared = reduce(q ** 2, 'b h i d -> b h i 1', 'sum')
            k_squared = reduce(k ** 2, 'b h j d -> b h 1 j', 'sum')
            sim = sim * 2 - q_squared - k_squared

        # key padding mask
        if exists(mask):
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
