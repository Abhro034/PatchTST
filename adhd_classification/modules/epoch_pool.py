"""
Epoch Pooling Module

Fuses channel embeddings within each epoch to produce epoch-level embeddings.

Input:  [B, E, N, D] or [B, E, N, T, D] - channel embeddings or tokens
Output: [B, E, D] - epoch embeddings

where:
    B = batch size (subjects)
    E = epochs per subject
    N = number of channels (10 for PSG)
    D = embedding dimension
    T = number of tokens per channel (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class EpochPool(nn.Module):
    """
    Fuses channel embeddings within epochs.

    Supports multiple fusion strategies:
    - mean: Simple mean pooling across channels
    - attention: Attention-weighted pooling
    - concat: Concatenate + linear projection
    - graph: Uses GNN to fuse (for Step 4+)
    """
    def __init__(
        self,
        n_channels: int = 10,
        d_model: int = 128,
        pool_type: str = 'mean',
        use_tokens: bool = False,  # If True, expects [B, E, N, T, D]
        **kwargs
    ):
        """
        Args:
            n_channels: Number of channels (N)
            d_model: Embedding dimension (D)
            pool_type: Type of pooling ('mean', 'attention', 'concat', 'graph')
            use_tokens: If True, input has token dimension
            **kwargs: Additional arguments for specific poolers
        """
        super().__init__()

        self.n_channels = n_channels
        self.d_model = d_model
        self.pool_type = pool_type
        self.use_tokens = use_tokens

        if pool_type == 'mean':
            # Simple mean pooling - no parameters
            pass

        elif pool_type == 'attention':
            # Learnable attention weights
            self.attention = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1)
            )

        elif pool_type == 'concat':
            # Concatenate all channels and project
            self.projection = nn.Linear(n_channels * d_model, d_model)

        elif pool_type == 'graph':
            # Placeholder - will be replaced by GNN in Step 4
            # For now, use mean pooling
            print("Warning: 'graph' pool_type requires GNN module. Using mean pooling.")
            self.pool_type = 'mean'

        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

    def forward(self, h_chan):
        """
        Args:
            h_chan: [B, E, N, D] or [B, E, N, T, D] - channel embeddings or tokens
        Returns:
            u: [B, E, D] - epoch embeddings
        """
        if self.use_tokens:
            # h_chan: [B, E, N, T, D]
            # First pool across tokens within each channel
            h_chan = h_chan.mean(dim=3)  # [B, E, N, D]

        # Now h_chan: [B, E, N, D]
        B, E, N, D = h_chan.shape

        if self.pool_type == 'mean':
            # Simple mean pooling across channels
            u = h_chan.mean(dim=2)  # [B, E, D]

        elif self.pool_type == 'attention':
            # Attention-weighted pooling
            # Compute attention scores
            attn_scores = self.attention(h_chan)  # [B, E, N, 1]
            attn_weights = F.softmax(attn_scores, dim=2)  # [B, E, N, 1]

            # Weighted sum
            u = (h_chan * attn_weights).sum(dim=2)  # [B, E, D]

        elif self.pool_type == 'concat':
            # Concatenate all channels
            h_flat = h_chan.reshape(B, E, N * D)  # [B, E, N*D]
            u = self.projection(h_flat)  # [B, E, D]

        return u


class GraphEpochPool(nn.Module):
    """
    Graph-based epoch pooling for Step 4+.

    Uses GNN to update node features, then pools across nodes and tokens.
    """
    def __init__(
        self,
        n_channels: int = 10,
        d_model: int = 128,
        graph_learner: Optional[nn.Module] = None,
        gnn_block: Optional[nn.Module] = None,
        node_pool_type: str = 'mean',  # How to pool across nodes
        token_pool_type: str = 'mean',  # How to pool across tokens
        **kwargs
    ):
        """
        Args:
            n_channels: Number of channels (N)
            d_model: Embedding dimension (D)
            graph_learner: Module to compute adjacency matrix
            gnn_block: Graph neural network module
            node_pool_type: Pooling across nodes ('mean', 'max', 'sum')
            token_pool_type: Pooling across tokens ('mean', 'max', 'sum')
        """
        super().__init__()

        self.n_channels = n_channels
        self.d_model = d_model
        self.graph_learner = graph_learner
        self.gnn_block = gnn_block
        self.node_pool_type = node_pool_type
        self.token_pool_type = token_pool_type

        if graph_learner is None or gnn_block is None:
            raise ValueError("GraphEpochPool requires both graph_learner and gnn_block")

    def forward(self, H, return_intermediate=False):
        """
        Args:
            H: [B, E, N, T, D] - patch tokens for all channels
            return_intermediate: If True, return adjacency matrices and node features

        Returns:
            u: [B, E, D] - epoch embeddings
            intermediates: Dict with 'A' and 'Z' (if return_intermediate=True)
        """
        B, E, N, T, D = H.shape

        # Process each epoch and token independently
        # Reshape for batch processing
        H_reshape = H.permute(0, 1, 3, 2, 4).reshape(B * E * T, N, D)  # [B*E*T, N, D]

        # Learn adjacency matrices
        A = self.graph_learner(H_reshape)  # [B*E*T, N, N]

        # Apply GNN
        Z = self.gnn_block(H_reshape, A)  # [B*E*T, N, D]

        # Reshape back
        Z = Z.reshape(B, E, T, N, D)  # [B, E, T, N, D]

        # Pool across nodes
        if self.node_pool_type == 'mean':
            Z_node = Z.mean(dim=3)  # [B, E, T, D]
        elif self.node_pool_type == 'max':
            Z_node = Z.max(dim=3)[0]
        elif self.node_pool_type == 'sum':
            Z_node = Z.sum(dim=3)
        else:
            raise ValueError(f"Unknown node_pool_type: {self.node_pool_type}")

        # Pool across tokens
        if self.token_pool_type == 'mean':
            u = Z_node.mean(dim=2)  # [B, E, D]
        elif self.token_pool_type == 'max':
            u = Z_node.max(dim=2)[0]
        elif self.token_pool_type == 'sum':
            u = Z_node.sum(dim=2)
        else:
            raise ValueError(f"Unknown token_pool_type: {self.token_pool_type}")

        if return_intermediate:
            # Reshape A back to [B, E, T, N, N]
            A = A.reshape(B, E, T, N, N)
            intermediates = {
                'A': A,  # Adjacency matrices
                'Z': Z,  # Node features after GNN
            }
            return u, intermediates

        return u
