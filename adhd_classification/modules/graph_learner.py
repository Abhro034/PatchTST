"""
Graph Learner Module

Learns dynamic adjacency matrices for channel interactions.

Input:  [B, N, D] - node features (channel embeddings)
Output: [B, N, N] - adjacency matrices

where:
    B = batch size (can be B*E*T for processing all graphs)
    N = number of nodes (channels)
    D = feature dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


class GraphLearner(nn.Module):
    """
    Learns adjacency matrices from node features.

    Supports multiple strategies:
    - attention: Attention-based similarity
    - cosine: Cosine similarity
    - mlp: MLP-based similarity
    - knn: K-nearest neighbors with attention
    """
    def __init__(
        self,
        d_model: int = 128,
        learner_type: str = 'attention',
        **kwargs
    ):
        """
        Args:
            d_model: Feature dimension (D)
            learner_type: Type of graph learner ('attention', 'cosine', 'mlp', 'knn')
            **kwargs: Additional arguments for specific learners
        """
        super().__init__()

        self.d_model = d_model
        self.learner_type = learner_type

        if learner_type == 'attention':
            # Attention-based similarity
            self.query = nn.Linear(d_model, d_model)
            self.key = nn.Linear(d_model, d_model)
            self.scale = math.sqrt(d_model)

        elif learner_type == 'cosine':
            # Cosine similarity - no parameters needed
            # Optional: learnable temperature parameter
            self.temperature = nn.Parameter(torch.ones(1) * kwargs.get('init_temperature', 1.0))

        elif learner_type == 'mlp':
            # MLP-based similarity
            hidden_dim = kwargs.get('hidden_dim', d_model)
            self.mlp = nn.Sequential(
                nn.Linear(2 * d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        elif learner_type == 'knn':
            # K-nearest neighbors
            self.k = kwargs.get('k', 5)
            self.query = nn.Linear(d_model, d_model)
            self.key = nn.Linear(d_model, d_model)
            self.scale = math.sqrt(d_model)

        else:
            raise ValueError(f"Unknown learner_type: {learner_type}")

        # Common parameters
        self.sparsity_reg = kwargs.get('sparsity_reg', 0.0)
        self.edge_dropout = kwargs.get('edge_dropout', 0.0)

    def forward(self, x, return_raw=False):
        """
        Args:
            x: [B, N, D] - node features
            return_raw: If True, return raw scores before softmax

        Returns:
            A: [B, N, N] - adjacency matrices (row-normalized)
            raw_scores (optional): Raw similarity scores
        """
        B, N, D = x.shape

        if self.learner_type == 'attention':
            # Attention-based similarity
            Q = self.query(x)  # [B, N, D]
            K = self.key(x)    # [B, N, D]

            # Compute similarity
            scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, N, N]

        elif self.learner_type == 'cosine':
            # Cosine similarity
            # Normalize features
            x_norm = F.normalize(x, p=2, dim=-1)  # [B, N, D]

            # Compute cosine similarity
            scores = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [B, N, N]

            # Apply temperature
            scores = scores / self.temperature

        elif self.learner_type == 'mlp':
            # MLP-based similarity
            # Expand features for all pairs
            x_i = x.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
            x_j = x.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]

            # Concatenate pairs
            pairs = torch.cat([x_i, x_j], dim=-1)  # [B, N, N, 2D]

            # Compute scores
            scores = self.mlp(pairs).squeeze(-1)  # [B, N, N]

        elif self.learner_type == 'knn':
            # K-nearest neighbors
            Q = self.query(x)  # [B, N, D]
            K = self.key(x)    # [B, N, D]

            # Compute similarity
            scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, N, N]

            # Keep only top-k neighbors
            k = min(self.k, N - 1)  # Exclude self-loops
            topk_values, topk_indices = torch.topk(scores, k + 1, dim=-1)  # Include self

            # Create sparse adjacency
            scores_sparse = torch.full_like(scores, float('-inf'))
            for i in range(N):
                scores_sparse[:, i, :].scatter_(1, topk_indices[:, i, :], topk_values[:, i, :])

            scores = scores_sparse

        # Store raw scores if requested
        raw_scores = scores.clone() if return_raw else None

        # Apply softmax to get adjacency matrix (row-normalized)
        A = F.softmax(scores, dim=-1)  # [B, N, N]

        # Apply edge dropout during training
        if self.training and self.edge_dropout > 0:
            dropout_mask = torch.bernoulli(torch.ones_like(A) * (1 - self.edge_dropout))
            A = A * dropout_mask
            # Re-normalize rows
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

        if return_raw:
            return A, raw_scores
        return A

    def compute_sparsity_loss(self, A):
        """
        Computes sparsity regularization loss.

        Args:
            A: [B, N, N] - adjacency matrices

        Returns:
            loss: Scalar - sparsity loss
        """
        if self.sparsity_reg == 0:
            return torch.tensor(0.0, device=A.device)

        # L1 regularization on adjacency weights
        sparsity_loss = self.sparsity_reg * A.abs().mean()

        return sparsity_loss


class AdaptiveGraphLearner(nn.Module):
    """
    Advanced graph learner for Step 5 improvements.

    Combines multiple similarity metrics and learns to weight them.
    """
    def __init__(
        self,
        d_model: int = 128,
        use_attention: bool = True,
        use_cosine: bool = True,
        use_mlp: bool = False,
        **kwargs
    ):
        """
        Args:
            d_model: Feature dimension (D)
            use_attention: Use attention-based similarity
            use_cosine: Use cosine similarity
            use_mlp: Use MLP-based similarity
        """
        super().__init__()

        self.d_model = d_model
        self.metrics = []

        if use_attention:
            self.attention_learner = GraphLearner(d_model, 'attention', **kwargs)
            self.metrics.append('attention')

        if use_cosine:
            self.cosine_learner = GraphLearner(d_model, 'cosine', **kwargs)
            self.metrics.append('cosine')

        if use_mlp:
            self.mlp_learner = GraphLearner(d_model, 'mlp', **kwargs)
            self.metrics.append('mlp')

        n_metrics = len(self.metrics)
        if n_metrics == 0:
            raise ValueError("At least one metric must be enabled")

        # Learnable weights for combining metrics
        self.metric_weights = nn.Parameter(torch.ones(n_metrics) / n_metrics)

    def forward(self, x):
        """
        Args:
            x: [B, N, D] - node features

        Returns:
            A: [B, N, N] - adjacency matrices (weighted combination)
        """
        adjacencies = []

        if 'attention' in self.metrics:
            A_attn = self.attention_learner(x)
            adjacencies.append(A_attn)

        if 'cosine' in self.metrics:
            A_cos = self.cosine_learner(x)
            adjacencies.append(A_cos)

        if 'mlp' in self.metrics:
            A_mlp = self.mlp_learner(x)
            adjacencies.append(A_mlp)

        # Normalize weights
        weights = F.softmax(self.metric_weights, dim=0)

        # Weighted combination
        A = sum(w * A_m for w, A_m in zip(weights, adjacencies))

        # Re-normalize rows
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

        return A


class TwoHopGraphLearner(nn.Module):
    """
    Graph learner with two-hop connections for Step 5.

    Augments first-order adjacency with second-order connections.
    """
    def __init__(
        self,
        d_model: int = 128,
        base_learner_type: str = 'attention',
        two_hop_weight: float = 0.3,
        **kwargs
    ):
        """
        Args:
            d_model: Feature dimension (D)
            base_learner_type: Base graph learner type
            two_hop_weight: Weight for two-hop connections (0-1)
        """
        super().__init__()

        self.base_learner = GraphLearner(d_model, base_learner_type, **kwargs)
        self.two_hop_weight = two_hop_weight

    def forward(self, x):
        """
        Args:
            x: [B, N, D] - node features

        Returns:
            A: [B, N, N] - adjacency with two-hop connections
        """
        # Get base adjacency
        A1 = self.base_learner(x)  # [B, N, N]

        # Compute two-hop adjacency
        A2 = torch.bmm(A1, A1)  # [B, N, N]

        # Combine one-hop and two-hop
        A = (1 - self.two_hop_weight) * A1 + self.two_hop_weight * A2

        # Re-normalize rows
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

        return A
