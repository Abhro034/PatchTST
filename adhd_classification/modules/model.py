"""
ADHD Classification Model

Main model orchestrating all components for subject-level ADHD classification.

Input:  [B, E, N, L] - PSG data
        B = batch size (subjects)
        E = epochs per subject
        N = channels (10 for PSG)
        L = samples per epoch

Output: [B, 1] - ADHD probability
        Optional: Dict of intermediate tensors and auxiliary outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .temporal_encoder import TemporalEncoder
from .epoch_pool import EpochPool, GraphEpochPool
from .night_pool import NightPool, ChunkNightPool
from .graph_learner import GraphLearner, AdaptiveGraphLearner, TwoHopGraphLearner
from .gnn_block import GNNBlock, MultiScaleGNN


class ADHDClassifier(nn.Module):
    """
    Modular ADHD classification model.

    Supports progressive development from baseline to advanced graph models.
    """
    def __init__(
        self,
        # Data parameters
        n_channels: int = 10,
        seq_len: int = 3000,  # Samples per 30s epoch (e.g., 100 Hz)
        d_model: int = 128,

        # Step configuration
        step: int = 1,  # 1: TCN baseline, 2: PatchTST, 3: Tokens, 4: Graph, 5: Advanced

        # Temporal encoder parameters
        encoder_type: str = 'tcn',  # 'tcn' or 'patchtst'
        encoder_kwargs: Optional[Dict] = None,

        # Epoch pooling parameters
        epoch_pool_type: str = 'mean',  # 'mean', 'attention', 'concat'
        epoch_pool_kwargs: Optional[Dict] = None,

        # Night pooling parameters
        night_pool_type: str = 'mean',  # 'mean', 'max', 'attention', 'topk', 'stage_aware'
        night_pool_kwargs: Optional[Dict] = None,

        # Graph parameters (Step 4+)
        use_graph: bool = False,
        graph_learner_type: str = 'attention',
        graph_learner_kwargs: Optional[Dict] = None,
        gnn_type: str = 'gat',
        gnn_kwargs: Optional[Dict] = None,

        # Classification head
        head_dropout: float = 0.3,

        # Multi-task learning (Step 5)
        use_multitask: bool = False,
        multitask_kwargs: Optional[Dict] = None,

        **kwargs
    ):
        super().__init__()

        self.n_channels = n_channels
        self.seq_len = seq_len
        self.d_model = d_model
        self.step = step
        self.use_graph = use_graph
        self.use_multitask = use_multitask

        # Initialize kwargs
        encoder_kwargs = encoder_kwargs or {}
        epoch_pool_kwargs = epoch_pool_kwargs or {}
        night_pool_kwargs = night_pool_kwargs or {}
        graph_learner_kwargs = graph_learner_kwargs or {}
        gnn_kwargs = gnn_kwargs or {}
        multitask_kwargs = multitask_kwargs or {}

        # Determine if we're using token outputs (Step 3+)
        use_tokens = (step >= 3) or use_graph
        patch_pooling = not use_tokens

        # 1. Temporal Encoder
        self.temporal_encoder = TemporalEncoder(
            seq_len=seq_len,
            d_model=d_model,
            encoder_type=encoder_type,
            patch_pooling=patch_pooling,
            **encoder_kwargs
        )

        # 2. Epoch Pooling
        if use_graph:
            # Use graph-based epoch pooling
            graph_learner = self._build_graph_learner(
                d_model, graph_learner_type, graph_learner_kwargs
            )
            gnn_block = self._build_gnn_block(
                d_model, gnn_type, gnn_kwargs
            )

            self.epoch_pool = GraphEpochPool(
                n_channels=n_channels,
                d_model=d_model,
                graph_learner=graph_learner,
                gnn_block=gnn_block,
                **epoch_pool_kwargs
            )
        else:
            # Standard pooling
            self.epoch_pool = EpochPool(
                n_channels=n_channels,
                d_model=d_model,
                pool_type=epoch_pool_type,
                use_tokens=use_tokens,
                **epoch_pool_kwargs
            )

        # 3. Night Pooling
        if night_pool_type == 'chunk':
            self.night_pool = ChunkNightPool(
                d_model=d_model,
                **night_pool_kwargs
            )
        else:
            self.night_pool = NightPool(
                d_model=d_model,
                pool_type=night_pool_type,
                **night_pool_kwargs
            )

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model // 2, 1)
        )

        # 5. Multi-task Heads (Step 5)
        if use_multitask:
            self.multitask_heads = self._build_multitask_heads(
                d_model, multitask_kwargs
            )

    def _build_graph_learner(self, d_model, learner_type, kwargs):
        """Build graph learner module."""
        if learner_type == 'adaptive':
            return AdaptiveGraphLearner(d_model, **kwargs)
        elif learner_type == 'twohop':
            return TwoHopGraphLearner(d_model, **kwargs)
        else:
            return GraphLearner(d_model, learner_type, **kwargs)

    def _build_gnn_block(self, d_model, gnn_type, kwargs):
        """Build GNN block module."""
        if gnn_type == 'multiscale':
            return MultiScaleGNN(d_model, **kwargs)
        else:
            return GNNBlock(d_model, gnn_type, **kwargs)

    def _build_multitask_heads(self, d_model, kwargs):
        """Build multi-task heads."""
        heads = nn.ModuleDict()

        if kwargs.get('sleep_staging', False):
            n_stages = kwargs.get('n_stages', 5)
            heads['sleep_stage'] = nn.Linear(d_model, n_stages)

        if kwargs.get('arousal_detection', False):
            heads['arousal'] = nn.Linear(d_model, 1)

        if kwargs.get('respiratory_events', False):
            heads['respiratory'] = nn.Linear(d_model, 1)

        if kwargs.get('hrv_estimation', False):
            heads['hrv'] = nn.Linear(d_model, 1)

        return heads

    def forward(self, x, stage_labels=None, attention_mask=None, return_intermediates=False):
        """
        Forward pass.

        Args:
            x: [B, E, N, L] - PSG data
            stage_labels: [B, E] - sleep stage labels (optional)
            attention_mask: [B, E] - mask for padded epochs (1=real, 0=padding)
            return_intermediates: If True, return intermediate tensors

        Returns:
            output: Dict containing:
                - 'logits': [B, 1] - ADHD logits
                - 'intermediates': Dict of intermediate tensors (if requested)
                - 'multitask': Dict of auxiliary outputs (if enabled)
        """
        B, E, N, L = x.shape
        intermediates = {}

        # 1. Temporal Encoding
        # Reshape to [B*E*N, L]
        x_flat = x.reshape(B * E * N, L)

        # Encode
        h = self.temporal_encoder(x_flat)  # [B*E*N, D] or [B*E*N, T, D]

        if len(h.shape) == 2:
            # Channel embeddings: [B*E*N, D]
            h_chan = h.reshape(B, E, N, self.d_model)  # [B, E, N, D]
            intermediates['H'] = h_chan
        else:
            # Token outputs: [B*E*N, T, D]
            T = h.shape[1]
            h_tokens = h.reshape(B, E, N, T, self.d_model)  # [B, E, N, T, D]
            intermediates['H'] = h_tokens

        # 2. Epoch Pooling
        if self.use_graph:
            # Graph-based pooling
            u, graph_intermediates = self.epoch_pool(
                intermediates['H'], return_intermediate=True
            )
            intermediates.update(graph_intermediates)
        else:
            # Standard pooling
            u = self.epoch_pool(intermediates['H'])  # [B, E, D]

        intermediates['u'] = u

        # 3. Night Pooling (with attention mask for padding)
        if hasattr(self.night_pool, 'forward') and 'stage_labels' in self.night_pool.forward.__code__.co_varnames:
            # Stage-aware pooling (with mask)
            z = self.night_pool(u, stage_labels=stage_labels, attention_mask=attention_mask, return_attention=False)
        elif hasattr(self.night_pool, 'forward') and 'attention_mask' in self.night_pool.forward.__code__.co_varnames:
            # Regular pooling with mask support
            z = self.night_pool(u, attention_mask=attention_mask)
        else:
            z = self.night_pool(u)  # [B, D]

        intermediates['z'] = z

        # 4. Classification
        logits = self.classifier(z)  # [B, 1]

        # 5. Multi-task outputs
        multitask_outputs = {}
        if self.use_multitask:
            # Apply multi-task heads to epoch embeddings
            for task_name, head in self.multitask_heads.items():
                if task_name == 'sleep_stage':
                    # Per-epoch prediction
                    task_out = head(u)  # [B, E, n_stages]
                else:
                    # Per-epoch binary prediction
                    task_out = head(u)  # [B, E, 1]

                multitask_outputs[task_name] = task_out

        # Prepare output
        output = {
            'logits': logits,
        }

        if return_intermediates:
            output['intermediates'] = intermediates

        if self.use_multitask:
            output['multitask'] = multitask_outputs

        return output

    def get_num_params(self):
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ADHDClassifierWithConfounds(nn.Module):
    """
    ADHD Classifier with confound control (Step 5).

    Uses adversarial learning to remove confound information.
    """
    def __init__(
        self,
        base_model_kwargs: Dict,
        confound_dims: Dict,  # e.g., {'age': 1, 'sex': 2, 'ahi': 1}
        adversarial_weight: float = 0.1,
        **kwargs
    ):
        """
        Args:
            base_model_kwargs: Kwargs for base ADHDClassifier
            confound_dims: Dict mapping confound names to output dimensions
            adversarial_weight: Weight for adversarial loss
        """
        super().__init__()

        self.base_model = ADHDClassifier(**base_model_kwargs)
        self.confound_dims = confound_dims
        self.adversarial_weight = adversarial_weight

        # Confound prediction heads
        d_model = base_model_kwargs.get('d_model', 128)
        self.confound_heads = nn.ModuleDict()

        for confound_name, dim in confound_dims.items():
            self.confound_heads[confound_name] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, dim)
            )

    def forward(self, x, confounds=None, return_intermediates=False):
        """
        Forward pass with confound prediction.

        Args:
            x: [B, E, N, L] - PSG data
            confounds: Dict of confound values (for computing adversarial loss)
            return_intermediates: If True, return intermediate tensors

        Returns:
            output: Dict containing:
                - 'logits': [B, 1] - ADHD logits
                - 'confound_preds': Dict of confound predictions
                - 'intermediates': Dict of intermediate tensors (if requested)
        """
        # Forward through base model
        output = self.base_model(x, return_intermediates=True)

        # Get night embedding
        z = output['intermediates']['z']  # [B, D]

        # Predict confounds from night embedding
        confound_preds = {}
        for confound_name, head in self.confound_heads.items():
            confound_preds[confound_name] = head(z)

        output['confound_preds'] = confound_preds

        if not return_intermediates:
            del output['intermediates']

        return output

    def compute_adversarial_loss(self, output, confounds):
        """
        Compute adversarial loss for confound removal.

        Args:
            output: Model output dict
            confounds: Dict of ground-truth confounds

        Returns:
            loss: Adversarial loss (to be minimized by classifier, maximized by confound predictors)
        """
        loss = 0.0

        for confound_name, pred in output['confound_preds'].items():
            if confound_name in confounds:
                target = confounds[confound_name]

                if pred.shape[-1] == 1:
                    # Regression (e.g., age, AHI)
                    loss += F.mse_loss(pred.squeeze(-1), target)
                else:
                    # Classification (e.g., sex, site)
                    loss += F.cross_entropy(pred, target)

        return self.adversarial_weight * loss
