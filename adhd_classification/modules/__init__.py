"""
ADHD Classification Pipeline Modules

This package contains modular components for subject-level ADHD classification
using overnight polysomnography (PSG) data.

Modules:
    - temporal_encoder: Encodes time-series data (TCN or PatchTST)
    - epoch_pool: Fuses channel embeddings within epochs
    - night_pool: Aggregates epoch embeddings across the night (MIL)
    - graph_learner: Learns dynamic adjacency matrices
    - gnn_block: Graph neural network for node updates
    - model: Main classification model orchestrating all components
"""

from .temporal_encoder import TemporalEncoder
from .epoch_pool import EpochPool, GraphEpochPool
from .night_pool import NightPool, ChunkNightPool
from .graph_learner import GraphLearner, AdaptiveGraphLearner, TwoHopGraphLearner
from .gnn_block import GNNBlock, MultiScaleGNN
from .model import ADHDClassifier, ADHDClassifierWithConfounds

__all__ = [
    'TemporalEncoder',
    'EpochPool',
    'GraphEpochPool',
    'NightPool',
    'ChunkNightPool',
    'GraphLearner',
    'AdaptiveGraphLearner',
    'TwoHopGraphLearner',
    'GNNBlock',
    'MultiScaleGNN',
    'ADHDClassifier',
    'ADHDClassifierWithConfounds'
]
