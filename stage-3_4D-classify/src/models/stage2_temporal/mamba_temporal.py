"""
Temporal encoder using stacked Mamba blocks.

Processes sequence of frame embeddings for classification.
"""

import torch
import torch.nn as nn
from typing import Optional

from .mamba_block import MambaBlock, MixerModel


class TemporalMambaEncoder(nn.Module):
    """
    Stage 2: Mamba-based temporal encoder.

    Processes sequence of frame embeddings and outputs sequence-level representation
    for classification.

    Architecture:
        Input projection -> Stacked Mamba blocks -> Output projection -> Aggregation
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        use_mixer_model: bool = True,
        aggregation: str = 'mean',  # 'mean', 'last', 'max', 'attention'
    ):
        """
        Args:
            input_dim: Dimension of input frame embeddings
            hidden_dim: Hidden dimension of Mamba blocks
            output_dim: Output dimension
            num_layers: Number of stacked Mamba blocks
            d_state: SSM state dimension
            d_conv: Depthwise conv kernel size
            expand: Expansion factor
            dropout: Dropout rate
            bidirectional: If True, process sequence in both directions
            use_mixer_model: If True, use MixerModel instead of simple MambaBlocks
            aggregation: How to aggregate sequence to single embedding
        """
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        if use_mixer_model:
            # Use the full MixerModel implementation
            self.forward_layers = MixerModel(
                d_model=hidden_dim,
                n_layer=num_layers,
                drop_out_in_block=dropout,
                drop_path_rate=dropout,
            )
        else:
            # Use simpler stacked MambaBlocks
            self.forward_layers = nn.ModuleList([
                MambaBlock(
                    dim=hidden_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])

        self.use_mixer_model = use_mixer_model

        # Optional bidirectional processing
        self.bidirectional = bidirectional
        if bidirectional:
            if use_mixer_model:
                self.reverse_layers = MixerModel(
                    d_model=hidden_dim,
                    n_layer=num_layers,
                    drop_out_in_block=dropout,
                    drop_path_rate=dropout,
                )
            else:
                self.reverse_layers = nn.ModuleList([
                    MambaBlock(
                        dim=hidden_dim,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ])

        # Output projection
        out_channels = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Linear(out_channels, output_dim)

        self.norm = nn.LayerNorm(output_dim)

        # Aggregation method
        self.aggregation = aggregation
        if aggregation == 'attention':
            self.attn_pool = nn.Sequential(
                nn.Linear(output_dim, output_dim // 2),
                nn.Tanh(),
                nn.Linear(output_dim // 2, 1),
            )

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            frame_embeddings: (B, T, D) sequence of frame embeddings
            mask: (B, T) optional mask for valid frames

        Returns:
            output: (B, output_dim) sequence-level embedding
        """
        x = self.input_proj(frame_embeddings)

        # Forward pass through Mamba layers
        if self.use_mixer_model:
            x = self.forward_layers(x)
        else:
            for layer in self.forward_layers:
                x = layer(x)

        if self.bidirectional:
            # Reverse sequence processing
            x_rev = torch.flip(frame_embeddings, dims=[1])
            x_rev = self.input_proj(x_rev)

            if self.use_mixer_model:
                x_rev = self.reverse_layers(x_rev)
            else:
                for layer in self.reverse_layers:
                    x_rev = layer(x_rev)

            x_rev = torch.flip(x_rev, dims=[1])
            x = torch.cat([x, x_rev], dim=-1)

        x = self.output_proj(x)
        x = self.norm(x)

        # Aggregate sequence to single embedding
        output = self._aggregate(x, mask)

        return output

    def _aggregate(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate sequence to single embedding.

        Args:
            x: (B, T, D) sequence
            mask: (B, T) optional mask

        Returns:
            (B, D) aggregated embedding
        """
        if self.aggregation == 'last':
            if mask is not None:
                # Get last valid position for each sequence
                lengths = mask.sum(dim=1).long() - 1
                batch_idx = torch.arange(x.size(0), device=x.device)
                output = x[batch_idx, lengths]
            else:
                output = x[:, -1]

        elif self.aggregation == 'max':
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            output = x.max(dim=1)[0]

        elif self.aggregation == 'attention':
            # Attention-based pooling
            attn_logits = self.attn_pool(x).squeeze(-1)  # (B, T)
            if mask is not None:
                attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
            attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, T)
            output = torch.einsum('bt,btd->bd', attn_weights, x)

        else:  # 'mean'
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                output = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                output = x.mean(dim=1)

        return output

    def forward_with_intermediates(
        self,
        frame_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Forward pass returning intermediate representations.

        Args:
            frame_embeddings: (B, T, D) sequence of frame embeddings
            mask: (B, T) optional mask

        Returns:
            output: (B, output_dim) aggregated embedding
            sequence: (B, T, output_dim) full sequence output
        """
        x = self.input_proj(frame_embeddings)

        if self.use_mixer_model:
            x = self.forward_layers(x)
        else:
            for layer in self.forward_layers:
                x = layer(x)

        if self.bidirectional:
            x_rev = torch.flip(frame_embeddings, dims=[1])
            x_rev = self.input_proj(x_rev)

            if self.use_mixer_model:
                x_rev = self.reverse_layers(x_rev)
            else:
                for layer in self.reverse_layers:
                    x_rev = layer(x_rev)

            x_rev = torch.flip(x_rev, dims=[1])
            x = torch.cat([x, x_rev], dim=-1)

        x = self.output_proj(x)
        x = self.norm(x)

        output = self._aggregate(x, mask)

        return output, x
