"""
Night Pooling Module

Aggregates epoch embeddings across the night using Multiple Instance Learning (MIL).

Input:  [B, E, D] - epoch embeddings
        Optional: [B, E] - sleep stage labels
Output: [B, D] - night-level embeddings

where:
    B = batch size (subjects)
    E = epochs per subject
    D = embedding dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class NightPool(nn.Module):
    """
    Aggregates epoch embeddings into night-level representations.

    Supports multiple MIL strategies:
    - mean: Simple mean pooling
    - max: Max pooling
    - attention: Attention-based MIL
    - topk: Top-k pooling based on attention scores
    - stage_aware: Different pooling for different sleep stages
    """
    def __init__(
        self,
        d_model: int = 128,
        pool_type: str = 'mean',
        **kwargs
    ):
        """
        Args:
            d_model: Embedding dimension (D)
            pool_type: Type of pooling ('mean', 'max', 'attention', 'topk', 'stage_aware')
            **kwargs: Additional arguments for specific poolers
        """
        super().__init__()

        self.d_model = d_model
        self.pool_type = pool_type

        if pool_type == 'mean':
            # Simple mean pooling - no parameters
            pass

        elif pool_type == 'max':
            # Max pooling - no parameters
            pass

        elif pool_type == 'attention':
            # Attention-based MIL
            self.attention = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1)
            )

        elif pool_type == 'topk':
            # Top-k pooling
            self.k = kwargs.get('k', 10)  # Number of top epochs to keep
            self.attention = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1)
            )

        elif pool_type == 'stage_aware':
            # Stage-aware pooling - requires sleep stage labels
            self.n_stages = kwargs.get('n_stages', 5)  # W, N1, N2, N3, REM
            self.stage_pools = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.Tanh(),
                    nn.Linear(d_model // 2, 1)
                ) for _ in range(self.n_stages)
            ])

        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

    def forward(self, u, stage_labels=None, attention_mask=None, return_attention=False):
        """
        Args:
            u: [B, E, D] - epoch embeddings
            stage_labels: [B, E] - sleep stage labels (optional, for stage_aware)
            attention_mask: [B, E] - mask for padded epochs (1=real, 0=padding)
            return_attention: If True, return attention weights

        Returns:
            z: [B, D] - night-level embeddings
            alpha (optional): Attention weights if return_attention=True
        """
        B, E, D = u.shape

        # Convert attention_mask to boolean if provided
        if attention_mask is not None:
            mask_bool = attention_mask.bool()  # [B, E]
        else:
            mask_bool = None

        if self.pool_type == 'mean':
            # Masked mean pooling across epochs
            if attention_mask is not None:
                # Expand mask to match embedding dimension
                mask_expanded = attention_mask.unsqueeze(-1)  # [B, E, 1]
                # Masked sum
                u_masked = u * mask_expanded  # [B, E, D]
                z = u_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)  # [B, D]
            else:
                z = u.mean(dim=1)  # [B, D]
            alpha = None

        elif self.pool_type == 'max':
            # Masked max pooling across epochs
            if attention_mask is not None:
                # Set padded positions to -inf before max
                mask_expanded = attention_mask.unsqueeze(-1)  # [B, E, 1]
                u_masked = u.masked_fill(mask_expanded == 0, float('-inf'))
                z = u_masked.max(dim=1)[0]  # [B, D]
                # Replace -inf with 0 if all epochs were masked (shouldn't happen)
                z = torch.where(torch.isinf(z), torch.zeros_like(z), z)
            else:
                z = u.max(dim=1)[0]  # [B, D]
            alpha = None

        elif self.pool_type == 'attention':
            # Masked attention-weighted pooling
            # Compute attention scores
            attn_scores = self.attention(u)  # [B, E, 1]

            # Mask padding before softmax
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(~mask_bool.unsqueeze(-1), float('-inf'))

            alpha = F.softmax(attn_scores, dim=1)  # [B, E, 1]

            # Weighted sum
            z = (u * alpha).sum(dim=1)  # [B, D]

        elif self.pool_type == 'topk':
            # Masked top-k pooling
            # Compute attention scores
            attn_scores = self.attention(u).squeeze(-1)  # [B, E]

            # Mask padding before top-k selection
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(~mask_bool, float('-inf'))

            # Get top-k indices
            k = min(self.k, E)
            topk_values, topk_indices = torch.topk(attn_scores, k, dim=1)  # [B, k]

            # Normalize top-k scores
            alpha = F.softmax(topk_values, dim=1)  # [B, k]

            # Gather top-k embeddings
            # Expand indices for gathering
            topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, k, D]
            u_topk = torch.gather(u, 1, topk_indices_expanded)  # [B, k, D]

            # Weighted sum of top-k
            z = (u_topk * alpha.unsqueeze(-1)).sum(dim=1)  # [B, D]

            # Create full alpha for consistency
            if return_attention:
                full_alpha = torch.zeros(B, E, device=u.device)
                full_alpha.scatter_(1, topk_indices, alpha)
                alpha = full_alpha.unsqueeze(-1)  # [B, E, 1]

        elif self.pool_type == 'stage_aware':
            # Masked stage-aware pooling
            if stage_labels is None:
                raise ValueError("stage_aware pooling requires stage_labels")

            # Pool each stage separately
            stage_embeds = []
            for stage_id in range(self.n_stages):
                # Mask for current stage
                stage_mask = (stage_labels == stage_id)  # [B, E]

                # Combine with attention mask if provided
                if attention_mask is not None:
                    stage_mask = stage_mask & mask_bool

                # Get epochs for this stage
                # Create expanded mask
                stage_mask_expanded = stage_mask.unsqueeze(-1).float()  # [B, E, 1]

                # Masked embeddings
                u_stage = u * stage_mask_expanded  # [B, E, D]

                # Compute attention for this stage
                attn_scores = self.stage_pools[stage_id](u_stage)  # [B, E, 1]

                # Mask out non-stage epochs before softmax
                attn_scores = attn_scores.masked_fill(~stage_mask.unsqueeze(-1), float('-inf'))

                # Softmax (will be 0 for masked positions after exp(-inf))
                stage_alpha = F.softmax(attn_scores, dim=1)  # [B, E, 1]

                # Replace NaN with 0 (happens when no epochs for a stage)
                stage_alpha = torch.where(torch.isnan(stage_alpha), torch.zeros_like(stage_alpha), stage_alpha)

                # Weighted sum for this stage
                z_stage = (u * stage_alpha).sum(dim=1)  # [B, D]

                stage_embeds.append(z_stage)

            # Concatenate stage embeddings
            z = torch.cat(stage_embeds, dim=-1)  # [B, n_stages * D]

            # Project back to D dimensions
            if not hasattr(self, 'stage_projection'):
                self.stage_projection = nn.Linear(self.n_stages * D, D).to(u.device)

            z = self.stage_projection(z)  # [B, D]
            alpha = None  # Stage-aware doesn't return single alpha

        if return_attention:
            return z, alpha
        return z


class ChunkNightPool(nn.Module):
    """
    Chunk-based night pooling for Step 5 improvements.

    Divides the night into chunks (e.g., early, middle, late) and pools separately.
    """
    def __init__(
        self,
        d_model: int = 128,
        n_chunks: int = 3,
        chunk_pool_type: str = 'attention',
        **kwargs
    ):
        """
        Args:
            d_model: Embedding dimension (D)
            n_chunks: Number of chunks to divide the night into
            chunk_pool_type: Pooling within each chunk ('mean', 'attention')
        """
        super().__init__()

        self.d_model = d_model
        self.n_chunks = n_chunks
        self.chunk_pool_type = chunk_pool_type

        if chunk_pool_type == 'attention':
            self.chunk_attentions = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.Tanh(),
                    nn.Linear(d_model // 2, 1)
                ) for _ in range(n_chunks)
            ])

        # Final projection from concatenated chunks
        self.projection = nn.Linear(n_chunks * d_model, d_model)

    def forward(self, u):
        """
        Args:
            u: [B, E, D] - epoch embeddings

        Returns:
            z: [B, D] - night-level embeddings
        """
        B, E, D = u.shape

        # Divide epochs into chunks
        chunk_size = E // self.n_chunks
        chunk_embeds = []

        for i in range(self.n_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self.n_chunks - 1 else E

            u_chunk = u[:, start_idx:end_idx, :]  # [B, chunk_len, D]

            if self.chunk_pool_type == 'mean':
                z_chunk = u_chunk.mean(dim=1)  # [B, D]

            elif self.chunk_pool_type == 'attention':
                attn_scores = self.chunk_attentions[i](u_chunk)  # [B, chunk_len, 1]
                alpha = F.softmax(attn_scores, dim=1)  # [B, chunk_len, 1]
                z_chunk = (u_chunk * alpha).sum(dim=1)  # [B, D]

            chunk_embeds.append(z_chunk)

        # Concatenate chunks
        z = torch.cat(chunk_embeds, dim=-1)  # [B, n_chunks * D]

        # Project back to D
        z = self.projection(z)  # [B, D]

        return z
