"""
Temporal Encoder Module

Encodes time-series data from individual channels into embeddings.
Supports multiple backends: TCN (baseline) and PatchTST (advanced).

Input:  [B * E * N, L] - flattened channel sequences
Output: [B * E * N, D] - channel embeddings (patch_pooling=True)
        [B * E * N, T, D] - patch tokens (patch_pooling=False)

where:
    B = batch size (subjects)
    E = epochs per subject
    N = number of channels (10 for PSG)
    L = samples per epoch (after resampling)
    D = embedding dimension
    T = number of patch tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import sys
import os

# Add PatchTST to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../PatchTST_supervised'))
from layers.PatchTST_backbone import TSTiEncoder


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) baseline.

    Uses dilated causal convolutions to capture long-range dependencies.
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        num_channels: list = [64, 128, 128],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.d_model = d_model

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)
        self.projection = nn.Linear(num_channels[-1], d_model)

    def forward(self, x):
        """
        Args:
            x: [B*E*N, 1, L] - input sequences
        Returns:
            [B*E*N, D] - channel embeddings
        """
        # TCN expects [batch, channels, length]
        y = self.network(x)  # [B*E*N, num_channels[-1], L]

        # Global average pooling across time
        y = y.mean(dim=-1)  # [B*E*N, num_channels[-1]]

        # Project to d_model
        y = self.projection(y)  # [B*E*N, D]

        return y


class TemporalBlock(nn.Module):
    """
    Single temporal block with dilated convolutions, weight normalization, and residual connection.
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size,
                     stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size,
                     stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Removes padding from the end of the sequence to ensure causality."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class PatchTSTEncoder(nn.Module):
    """
    PatchTST-based encoder.

    Segments sequences into patches and uses transformer layers.
    Can output either pooled embeddings or token-level features.
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int = 128,
        patch_len: int = 25,  # 0.5-1s at typical PSG sampling rates
        stride: int = 12,     # 50% overlap
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 256,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        patch_pooling: bool = True,  # If True, pool patches to get epoch embedding
        **kwargs
    ):
        super().__init__()

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.patch_pooling = patch_pooling

        # Calculate number of patches
        self.patch_num = int((seq_len - patch_len) / stride + 1)

        # Use PatchTST's TSTiEncoder but for single channel
        # We'll process channels independently, so c_in=1
        self.encoder = TSTiEncoder(
            c_in=1,
            patch_num=self.patch_num,
            patch_len=patch_len,
            max_seq_len=1024,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=None,
            d_v=None,
            d_ff=d_ff,
            norm='BatchNorm',
            attn_dropout=attn_dropout,
            dropout=dropout,
            act="gelu",
            store_attn=False,
            key_padding_mask='auto',
            padding_var=None,
            attn_mask=None,
            res_attention=True,
            pre_norm=False,
            pe='zeros',
            learn_pe=True,
            verbose=False
        )

        if patch_pooling:
            # Pool across patches to get channel embedding
            self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: [B*E*N, 1, L] - input sequences with channel dimension
        Returns:
            [B*E*N, D] if patch_pooling=True
            [B*E*N, T, D] if patch_pooling=False
        """
        batch_size = x.shape[0]

        # Unfold into patches
        # x: [B*E*N, 1, L]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B*E*N, 1, patch_num, patch_len]
        x = x.permute(0, 1, 3, 2)  # [B*E*N, 1, patch_len, patch_num]

        # Pass through encoder
        z = self.encoder(x)  # [B*E*N, 1, d_model, patch_num]

        # Remove channel dimension (it's 1)
        z = z.squeeze(1)  # [B*E*N, d_model, patch_num]

        if self.patch_pooling:
            # Pool across patches
            z = self.pool(z).squeeze(-1)  # [B*E*N, d_model]
            return z
        else:
            # Return token-level outputs
            z = z.permute(0, 2, 1)  # [B*E*N, patch_num, d_model]
            return z


class TemporalEncoder(nn.Module):
    """
    Unified interface for temporal encoding.

    Supports multiple backends with the same API.
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int = 128,
        encoder_type: str = 'tcn',  # 'tcn' or 'patchtst'
        patch_pooling: bool = True,
        **kwargs
    ):
        """
        Args:
            seq_len: Length of input sequence (L)
            d_model: Embedding dimension (D)
            encoder_type: Type of encoder ('tcn' or 'patchtst')
            patch_pooling: If True, pool patches (Step 1-2). If False, return tokens (Step 3+)
            **kwargs: Additional arguments for specific encoders
        """
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.encoder_type = encoder_type
        self.patch_pooling = patch_pooling

        if encoder_type == 'tcn':
            self.encoder = TemporalConvNet(
                input_size=1,  # Single channel input
                d_model=d_model,
                num_channels=kwargs.get('num_channels', [64, 128, 128]),
                kernel_size=kwargs.get('kernel_size', 3),
                dropout=kwargs.get('dropout', 0.2)
            )
            self.num_tokens = None  # TCN doesn't produce tokens

        elif encoder_type == 'patchtst':
            patch_len = kwargs.get('patch_len', 25)
            stride = kwargs.get('stride', 12)

            self.encoder = PatchTSTEncoder(
                seq_len=seq_len,
                d_model=d_model,
                patch_len=patch_len,
                stride=stride,
                n_layers=kwargs.get('n_layers', 4),
                n_heads=kwargs.get('n_heads', 8),
                d_ff=kwargs.get('d_ff', 256),
                dropout=kwargs.get('dropout', 0.1),
                attn_dropout=kwargs.get('attn_dropout', 0.0),
                patch_pooling=patch_pooling
            )
            self.num_tokens = self.encoder.patch_num if not patch_pooling else None

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, x):
        """
        Args:
            x: [B*E*N, L] - flattened channel sequences
        Returns:
            [B*E*N, D] if patch_pooling=True
            [B*E*N, T, D] if patch_pooling=False
        """
        # Add channel dimension for conv layers
        x = x.unsqueeze(1)  # [B*E*N, 1, L]

        # Encode
        z = self.encoder(x)

        return z

    def get_output_shape(self):
        """Returns the output shape for debugging."""
        if self.patch_pooling:
            return f"[B*E*N, {self.d_model}]"
        else:
            return f"[B*E*N, {self.num_tokens}, {self.d_model}]"
