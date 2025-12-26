"""
Graph Neural Network Block Module

Updates node features via message passing on learned graphs.

Input:  [B, N, D] - node features
        [B, N, N] - adjacency matrix
Output: [B, N, D] - updated node features

where:
    B = batch size
    N = number of nodes (channels)
    D = feature dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


class GNNBlock(nn.Module):
    """
    Graph neural network block.

    Supports multiple GNN architectures:
    - gcn: Graph Convolutional Network
    - gat: Graph Attention Network
    - gin: Graph Isomorphism Network
    """
    def __init__(
        self,
        d_model: int = 128,
        gnn_type: str = 'gat',
        n_layers: int = 2,
        **kwargs
    ):
        """
        Args:
            d_model: Feature dimension (D)
            gnn_type: Type of GNN ('gcn', 'gat', 'gin')
            n_layers: Number of GNN layers
            **kwargs: Additional arguments for specific GNN types
        """
        super().__init__()

        self.d_model = d_model
        self.gnn_type = gnn_type
        self.n_layers = n_layers

        # Build layers
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            if gnn_type == 'gcn':
                layer = GCNLayer(d_model, d_model, **kwargs)
            elif gnn_type == 'gat':
                layer = GATLayer(d_model, d_model, **kwargs)
            elif gnn_type == 'gin':
                layer = GINLayer(d_model, d_model, **kwargs)
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")

            self.layers.append(layer)

    def forward(self, x, A):
        """
        Args:
            x: [B, N, D] - node features
            A: [B, N, N] - adjacency matrix

        Returns:
            z: [B, N, D] - updated node features
        """
        z = x

        for layer in self.layers:
            z = layer(z, A)

        return z


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer.

    Implements: h_i' = σ(Σ_j A_ij W h_j + b)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        use_residual: bool = True,
        **kwargs
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_residual = use_residual

        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        if use_residual and in_features == out_features:
            self.residual = True
        else:
            self.residual = False

    def forward(self, x, A):
        """
        Args:
            x: [B, N, D_in]
            A: [B, N, N]

        Returns:
            [B, N, D_out]
        """
        # Message passing
        h = torch.bmm(A, x)  # [B, N, D_in]

        # Linear transformation
        h = self.linear(h)  # [B, N, D_out]

        # Dropout
        h = self.dropout(h)

        # Residual connection
        if self.residual:
            h = h + x

        # Activation
        h = self.activation(h)

        return h


class GATLayer(nn.Module):
    """
    Graph Attention Network layer.

    Uses multi-head attention over neighbors.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,  # LeakyReLU slope
        use_residual: bool = True,
        **kwargs
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.use_residual = use_residual

        assert out_features % n_heads == 0, "out_features must be divisible by n_heads"
        self.d_k = out_features // n_heads

        # Linear transformations for each head
        self.W = nn.Linear(in_features, out_features)

        # Attention parameters for each head
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * self.d_k))
        nn.init.xavier_uniform_(self.a)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        if use_residual and in_features == out_features:
            self.residual = True
        else:
            self.residual = False

    def forward(self, x, A):
        """
        Args:
            x: [B, N, D_in]
            A: [B, N, N] - adjacency matrix (can be used as mask)

        Returns:
            [B, N, D_out]
        """
        B, N, _ = x.shape

        # Linear transformation
        h = self.W(x)  # [B, N, D_out]

        # Reshape for multi-head attention
        h = h.view(B, N, self.n_heads, self.d_k)  # [B, N, n_heads, d_k]

        # Compute attention coefficients
        # Expand for all pairs
        h_i = h.unsqueeze(2).expand(B, N, N, self.n_heads, self.d_k)  # [B, N, N, n_heads, d_k]
        h_j = h.unsqueeze(1).expand(B, N, N, self.n_heads, self.d_k)  # [B, N, N, n_heads, d_k]

        # Concatenate for attention
        h_cat = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, n_heads, 2*d_k]

        # Compute attention scores
        e = (h_cat * self.a).sum(dim=-1)  # [B, N, N, n_heads]
        e = self.leakyrelu(e)

        # Use adjacency as mask (set to -inf where A is very small)
        mask = (A.unsqueeze(-1) < 1e-8)  # [B, N, N, 1]
        e = e.masked_fill(mask, float('-inf'))

        # Softmax to get attention weights
        alpha = F.softmax(e, dim=2)  # [B, N, N, n_heads]
        alpha = self.dropout(alpha)

        # Apply attention
        alpha = alpha.unsqueeze(-1)  # [B, N, N, n_heads, 1]
        h_j = h.unsqueeze(1).expand(B, N, N, self.n_heads, self.d_k)  # [B, N, N, n_heads, d_k]

        h_prime = (alpha * h_j).sum(dim=2)  # [B, N, n_heads, d_k]

        # Concatenate heads
        h_prime = h_prime.reshape(B, N, self.out_features)  # [B, N, D_out]

        # Residual connection
        if self.residual:
            h_prime = h_prime + x

        return h_prime


class GINLayer(nn.Module):
    """
    Graph Isomorphism Network layer.

    Implements: h_i' = MLP((1 + ε) h_i + Σ_j A_ij h_j)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_residual: bool = True,
        learn_eps: bool = True,
        **kwargs
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_residual = use_residual

        if hidden_dim is None:
            hidden_dim = out_features

        # Epsilon parameter
        if learn_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('eps', torch.zeros(1))

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if use_residual and in_features == out_features:
            self.residual = True
        else:
            self.residual = False

    def forward(self, x, A):
        """
        Args:
            x: [B, N, D_in]
            A: [B, N, N]

        Returns:
            [B, N, D_out]
        """
        # Aggregate neighbors
        neighbor_sum = torch.bmm(A, x)  # [B, N, D_in]

        # Add self-connection with epsilon
        h = (1 + self.eps) * x + neighbor_sum  # [B, N, D_in]

        # Apply MLP
        h = self.mlp(h)  # [B, N, D_out]

        # Residual connection
        if self.residual:
            h = h + x

        return h


class MultiScaleGNN(nn.Module):
    """
    Multi-scale GNN for Step 5 improvements.

    Combines information at different graph scales.
    """
    def __init__(
        self,
        d_model: int = 128,
        gnn_type: str = 'gat',
        scales: list = [1, 2],  # Different hop distances
        **kwargs
    ):
        """
        Args:
            d_model: Feature dimension
            gnn_type: Base GNN type
            scales: List of hop distances to consider
        """
        super().__init__()

        self.scales = scales
        self.gnns = nn.ModuleList()

        for scale in scales:
            gnn = GNNBlock(d_model, gnn_type, n_layers=scale, **kwargs)
            self.gnns.append(gnn)

        # Combine multi-scale features
        self.combine = nn.Linear(len(scales) * d_model, d_model)

    def forward(self, x, A):
        """
        Args:
            x: [B, N, D]
            A: [B, N, N]

        Returns:
            [B, N, D]
        """
        features = []

        for gnn in self.gnns:
            h = gnn(x, A)
            features.append(h)

        # Concatenate multi-scale features
        h_concat = torch.cat(features, dim=-1)  # [B, N, len(scales)*D]

        # Combine
        h = self.combine(h_concat)  # [B, N, D]

        return h
