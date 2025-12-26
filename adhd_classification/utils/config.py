"""
Configuration classes for ADHD classification models.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class ModelConfig:
    """
    Configuration for ADHDClassifier model.

    Organizes all hyperparameters for different pipeline steps.
    """

    # Data parameters
    n_channels: int = 10
    seq_len: int = 3000  # Samples per 30s epoch (100 Hz * 30s)
    d_model: int = 128

    # Pipeline step
    step: int = 1  # 1: TCN baseline, 2: PatchTST, 3: Tokens, 4: Graph, 5: Advanced

    # Temporal encoder
    encoder_type: str = 'tcn'  # 'tcn' or 'patchtst'
    encoder_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Epoch pooling
    epoch_pool_type: str = 'mean'  # 'mean', 'attention', 'concat'
    epoch_pool_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Night pooling
    night_pool_type: str = 'mean'  # 'mean', 'max', 'attention', 'topk', 'stage_aware'
    night_pool_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Graph parameters (Step 4+)
    use_graph: bool = False
    graph_learner_type: str = 'attention'
    graph_learner_kwargs: Dict[str, Any] = field(default_factory=dict)
    gnn_type: str = 'gat'
    gnn_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Classification head
    head_dropout: float = 0.3

    # Multi-task learning (Step 5)
    use_multitask: bool = False
    multitask_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def step1_tcn_baseline(cls):
        """Step 1: TCN baseline configuration."""
        return cls(
            step=1,
            encoder_type='tcn',
            encoder_kwargs={
                'num_channels': [64, 128, 128],
                'kernel_size': 3,
                'dropout': 0.2
            },
            epoch_pool_type='mean',
            night_pool_type='mean',
            use_graph=False,
            use_multitask=False
        )

    @classmethod
    def step2_patchtst(cls, seq_len=3000):
        """Step 2: PatchTST encoder configuration."""
        # Calculate patch parameters
        # For 100 Hz sampling, 30s epoch = 3000 samples
        # Use patches of ~0.5s (50 samples) with 50% overlap
        patch_len = max(25, seq_len // 60)  # ~0.5s
        stride = patch_len // 2  # 50% overlap

        return cls(
            step=2,
            encoder_type='patchtst',
            encoder_kwargs={
                'patch_len': patch_len,
                'stride': stride,
                'n_layers': 4,
                'n_heads': 8,
                'd_ff': 256,
                'dropout': 0.1,
                'attn_dropout': 0.0
            },
            epoch_pool_type='mean',
            night_pool_type='attention',
            night_pool_kwargs={},
            use_graph=False,
            use_multitask=False
        )

    @classmethod
    def step3_tokens(cls, seq_len=3000):
        """Step 3: Token-level outputs configuration."""
        patch_len = max(25, seq_len // 60)
        stride = patch_len // 2

        return cls(
            step=3,
            encoder_type='patchtst',
            encoder_kwargs={
                'patch_len': patch_len,
                'stride': stride,
                'n_layers': 4,
                'n_heads': 8,
                'd_ff': 256,
                'dropout': 0.1,
                'attn_dropout': 0.0
            },
            epoch_pool_type='attention',
            epoch_pool_kwargs={},
            night_pool_type='attention',
            night_pool_kwargs={},
            use_graph=False,
            use_multitask=False
        )

    @classmethod
    def step4_graph(cls, seq_len=3000):
        """Step 4: Graph neural network configuration."""
        patch_len = max(25, seq_len // 60)
        stride = patch_len // 2

        return cls(
            step=4,
            encoder_type='patchtst',
            encoder_kwargs={
                'patch_len': patch_len,
                'stride': stride,
                'n_layers': 4,
                'n_heads': 8,
                'd_ff': 256,
                'dropout': 0.1,
                'attn_dropout': 0.0
            },
            epoch_pool_type='graph',
            epoch_pool_kwargs={
                'node_pool_type': 'mean',
                'token_pool_type': 'mean'
            },
            night_pool_type='attention',
            night_pool_kwargs={},
            use_graph=True,
            graph_learner_type='attention',
            graph_learner_kwargs={
                'sparsity_reg': 0.01,
                'edge_dropout': 0.1
            },
            gnn_type='gat',
            gnn_kwargs={
                'n_layers': 2,
                'n_heads': 4,
                'dropout': 0.1
            },
            use_multitask=False
        )

    @classmethod
    def step5_advanced(cls, seq_len=3000):
        """Step 5: Advanced features configuration."""
        patch_len = max(25, seq_len // 60)
        stride = patch_len // 2

        return cls(
            step=5,
            encoder_type='patchtst',
            encoder_kwargs={
                'patch_len': patch_len,
                'stride': stride,
                'n_layers': 4,
                'n_heads': 8,
                'd_ff': 256,
                'dropout': 0.1,
                'attn_dropout': 0.0
            },
            epoch_pool_type='graph',
            epoch_pool_kwargs={
                'node_pool_type': 'mean',
                'token_pool_type': 'mean'
            },
            night_pool_type='topk',
            night_pool_kwargs={
                'k': 20  # Top 20 epochs
            },
            use_graph=True,
            graph_learner_type='twohop',
            graph_learner_kwargs={
                'base_learner_type': 'attention',
                'two_hop_weight': 0.3,
                'sparsity_reg': 0.01,
                'edge_dropout': 0.1
            },
            gnn_type='gat',
            gnn_kwargs={
                'n_layers': 2,
                'n_heads': 4,
                'dropout': 0.1
            },
            use_multitask=True,
            multitask_kwargs={
                'sleep_staging': True,
                'arousal_detection': True,
                'n_stages': 5
            }
        )

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'n_channels': self.n_channels,
            'seq_len': self.seq_len,
            'd_model': self.d_model,
            'step': self.step,
            'encoder_type': self.encoder_type,
            'encoder_kwargs': self.encoder_kwargs,
            'epoch_pool_type': self.epoch_pool_type,
            'epoch_pool_kwargs': self.epoch_pool_kwargs,
            'night_pool_type': self.night_pool_type,
            'night_pool_kwargs': self.night_pool_kwargs,
            'use_graph': self.use_graph,
            'graph_learner_type': self.graph_learner_type,
            'graph_learner_kwargs': self.graph_learner_kwargs,
            'gnn_type': self.gnn_type,
            'gnn_kwargs': self.gnn_kwargs,
            'head_dropout': self.head_dropout,
            'use_multitask': self.use_multitask,
            'multitask_kwargs': self.multitask_kwargs
        }


@dataclass
class TrainingConfig:
    """
    Configuration for training ADHD classifier.
    """

    # Training parameters
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Optimizer
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd'
    scheduler: Optional[str] = 'cosine'  # 'cosine', 'step', None

    # Loss weights
    classification_weight: float = 1.0
    multitask_weight: float = 0.3
    adversarial_weight: float = 0.1

    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_metric: str = 'val_auc'

    # Logging
    log_interval: int = 10
    eval_interval: int = 1

    # Checkpointing
    save_dir: str = 'checkpoints'
    save_best_only: bool = True

    # Data
    num_workers: int = 4
    pin_memory: bool = True

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'classification_weight': self.classification_weight,
            'multitask_weight': self.multitask_weight,
            'adversarial_weight': self.adversarial_weight,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_metric': self.early_stopping_metric,
            'log_interval': self.log_interval,
            'eval_interval': self.eval_interval,
            'save_dir': self.save_dir,
            'save_best_only': self.save_best_only,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory
        }
