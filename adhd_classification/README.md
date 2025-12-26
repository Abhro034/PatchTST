# ADHD Classification Pipeline using PatchTST

A modular deep-learning pipeline for subject-level ADHD classification using overnight polysomnography (PSG) data, built on the PatchTST architecture.

## Overview

This pipeline implements a progressive approach to ADHD classification, from simple baselines to advanced graph-based models. The modular design allows easy experimentation with different components while maintaining fixed interfaces.

### Key Features

- **Modular Architecture**: Swappable components with fixed APIs
- **Progressive Development**: Five implementation steps from baseline to advanced
- **Multiple Backends**: TCN and PatchTST temporal encoders
- **Graph Neural Networks**: Dynamic graph learning for channel interactions
- **Multi-task Learning**: Auxiliary tasks for improved representations
- **Confound Control**: Adversarial learning for removing confounds

## Pipeline Architecture

### Data Flow

```
Input: [B, E, N, L]
  ↓
TemporalEncoder (per channel)
  ↓
[B, E, N, D] or [B, E, N, T, D]
  ↓
EpochPool (channel fusion)
  ↓
[B, E, D]
  ↓
NightPool (MIL aggregation)
  ↓
[B, D]
  ↓
Classifier
  ↓
Output: [B, 1]
```

Where:
- `B` = Batch size (subjects)
- `E` = Epochs per subject (30-second windows)
- `N` = Number of channels (10 for PSG)
- `L` = Samples per epoch
- `D` = Embedding dimension
- `T` = Number of patch tokens

## Implementation Steps

### Step 1: TCN Baseline

Simple baseline using Temporal Convolutional Networks.

**Features:**
- TCN temporal encoder
- Mean pooling for channel fusion
- Mean pooling for epoch aggregation

**Configuration:**
```python
from utils.config import ModelConfig
config = ModelConfig.step1_tcn_baseline()
model = ADHDClassifier(**config.to_dict())
```

**Purpose:** Verify data pipeline and establish baseline performance.

### Step 2: PatchTST Encoder

Replace TCN with PatchTST while keeping the same API.

**Features:**
- PatchTST temporal encoder
- Patch-level processing (0.5-1s patches with 50% overlap)
- 4-6 transformer layers
- Pooled patch embeddings → channel embeddings

**Configuration:**
```python
config = ModelConfig.step2_patchtst(seq_len=3000)
model = ADHDClassifier(**config.to_dict())
```

**Purpose:** Improve temporal encoding with self-attention.

### Step 3: Token-Level Outputs

Expose patch tokens for finer-grained processing.

**Features:**
- Token-level outputs from PatchTST
- Attention-based pooling
- Prepares for graph learning

**Configuration:**
```python
config = ModelConfig.step3_tokens(seq_len=3000)
model = ADHDClassifier(**config.to_dict())
```

**Purpose:** Enable graph construction at token level.

### Step 4: Graph Neural Networks

Add dynamic graph learning for channel interactions.

**Features:**
- GraphLearner: Learns adjacency matrices
- GNNBlock: Message passing (GAT or GIN)
- Graph per token (~5s bins → 6 graphs per epoch)
- Pool across nodes and tokens

**Configuration:**
```python
config = ModelConfig.step4_graph(seq_len=3000)
model = ADHDClassifier(**config.to_dict())
```

**Purpose:** Model inter-channel dependencies explicitly.

### Step 5: Advanced Features

Iterative improvements on the graph model.

**Features:**
- Top-k epoch pooling
- Stage-aware pooling (if sleep stages available)
- Multi-task learning (sleep staging, arousals, etc.)
- Two-hop graph aggregation
- Adversarial confound removal

**Configuration:**
```python
config = ModelConfig.step5_advanced(seq_len=3000)
model = ADHDClassifier(**config.to_dict())
```

**Purpose:** Maximize performance with advanced techniques.

## Module Overview

### TemporalEncoder

Encodes time-series data from individual channels.

**Backends:**
- `tcn`: Temporal Convolutional Network
- `patchtst`: PatchTST transformer

**Interface:**
```python
encoder = TemporalEncoder(
    seq_len=3000,
    d_model=128,
    encoder_type='patchtst',
    patch_pooling=True  # False for token outputs
)
output = encoder(x)  # [B*E*N, L] → [B*E*N, D] or [B*E*N, T, D]
```

### EpochPool

Fuses channel embeddings within epochs.

**Methods:**
- `mean`: Simple mean pooling
- `attention`: Attention-weighted pooling
- `concat`: Concatenate + project
- `graph`: Graph neural network (requires GraphLearner + GNNBlock)

**Interface:**
```python
pool = EpochPool(
    n_channels=10,
    d_model=128,
    pool_type='attention'
)
u = pool(h_chan)  # [B, E, N, D] → [B, E, D]
```

### NightPool

Aggregates epoch embeddings using Multiple Instance Learning.

**Methods:**
- `mean`: Mean pooling
- `max`: Max pooling
- `attention`: Attention MIL
- `topk`: Top-k pooling
- `stage_aware`: Stage-specific pooling

**Interface:**
```python
pool = NightPool(
    d_model=128,
    pool_type='attention'
)
z = pool(u)  # [B, E, D] → [B, D]
```

### GraphLearner

Learns adjacency matrices from node features.

**Methods:**
- `attention`: Attention-based similarity
- `cosine`: Cosine similarity
- `mlp`: MLP-based similarity
- `knn`: K-nearest neighbors
- `twohop`: Two-hop aggregation

**Interface:**
```python
learner = GraphLearner(
    d_model=128,
    learner_type='attention'
)
A = learner(x)  # [B, N, D] → [B, N, N]
```

### GNNBlock

Graph neural network for message passing.

**Architectures:**
- `gcn`: Graph Convolutional Network
- `gat`: Graph Attention Network
- `gin`: Graph Isomorphism Network

**Interface:**
```python
gnn = GNNBlock(
    d_model=128,
    gnn_type='gat',
    n_layers=2
)
z = gnn(x, A)  # [B, N, D], [B, N, N] → [B, N, D]
```

## Usage

### Basic Training

```python
from modules.model import ADHDClassifier
from utils.config import ModelConfig, TrainingConfig
from utils.data import create_dummy_data, PSGDataset, create_dataloader

# Create dummy data for testing
data = create_dummy_data(
    n_subjects=100,
    n_epochs=200,
    n_channels=10,
    seq_len=3000
)

# Create dataset
dataset = PSGDataset(data['X'], data['y'])
loader = create_dataloader(dataset, batch_size=8)

# Create model
config = ModelConfig.step4_graph(seq_len=3000)
model = ADHDClassifier(**config.to_dict())

# Train
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for batch in loader:
    X, y = batch['X'], batch['y']

    output = model(X)
    loss = criterion(output['logits'].squeeze(-1), y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### With Intermediate Outputs

```python
# Get intermediate tensors for debugging
output = model(X, return_intermediates=True)

print("Logits:", output['logits'].shape)
print("Intermediates:", output['intermediates'].keys())

# Access specific tensors
H = output['intermediates']['H']  # Channel embeddings or tokens
A = output['intermediates']['A']  # Adjacency matrices (if using graph)
Z = output['intermediates']['Z']  # GNN outputs (if using graph)
u = output['intermediates']['u']  # Epoch embeddings
z = output['intermediates']['z']  # Night embedding
```

### Multi-task Learning

```python
config = ModelConfig.step5_advanced(seq_len=3000)
config.use_multitask = True
config.multitask_kwargs = {
    'sleep_staging': True,
    'arousal_detection': True,
    'n_stages': 5
}

model = ADHDClassifier(**config.to_dict())

output = model(X)

# Primary task
adhd_logits = output['logits']

# Auxiliary tasks
sleep_stages = output['multitask']['sleep_stage']  # [B, E, 5]
arousals = output['multitask']['arousal']  # [B, E, 1]
```

### Confound Control

```python
from modules.model import ADHDClassifierWithConfounds

base_config = ModelConfig.step4_graph(seq_len=3000).to_dict()

model = ADHDClassifierWithConfounds(
    base_model_kwargs=base_config,
    confound_dims={'age': 1, 'sex': 2, 'ahi': 1},
    adversarial_weight=0.1
)

# Forward pass
output = model(X, confounds=confounds_dict)

# Compute losses
adhd_loss = criterion(output['logits'].squeeze(-1), y)
adv_loss = model.compute_adversarial_loss(output, confounds_dict)

total_loss = adhd_loss + adv_loss
```

## Examples

Run the example scripts to see each step in action:

```bash
# Step 1: TCN baseline
cd examples
python train_baseline.py

# Step 2: PatchTST encoder
python train_patchtst.py

# Step 4: Graph-based model
python train_graph.py
```

## Model Configuration Reference

### Common Parameters

- `n_channels`: Number of PSG channels (default: 10)
- `seq_len`: Samples per epoch (default: 3000 for 100 Hz, 30s)
- `d_model`: Embedding dimension (default: 128)
- `head_dropout`: Dropout in classification head (default: 0.3)

### PatchTST Parameters

- `patch_len`: Length of each patch (default: ~0.5s)
- `stride`: Stride between patches (default: 50% overlap)
- `n_layers`: Number of transformer layers (default: 4)
- `n_heads`: Number of attention heads (default: 8)
- `d_ff`: Feed-forward dimension (default: 256)

### Graph Parameters

- `graph_learner_type`: Type of graph learner
- `gnn_type`: Type of GNN architecture
- `sparsity_reg`: L1 regularization on adjacency (default: 0.01)
- `edge_dropout`: Dropout on edges (default: 0.1)

## Design Principles

### Modularity

Each component has a fixed interface and can be swapped without modifying other parts:

```python
# Swap encoder
encoder = TemporalEncoder(..., encoder_type='tcn')  # or 'patchtst'

# Swap pooling
pool = EpochPool(..., pool_type='mean')  # or 'attention', 'concat'

# Swap GNN
gnn = GNNBlock(..., gnn_type='gat')  # or 'gcn', 'gin'
```

### One File Per Module

- `temporal_encoder.py`: All temporal encoding logic
- `epoch_pool.py`: All epoch pooling methods
- `night_pool.py`: All night pooling methods
- `graph_learner.py`: All graph learning methods
- `gnn_block.py`: All GNN architectures
- `model.py`: Orchestration only

### Fixed Interfaces

Never modify existing modules when experimenting. Create new variants instead:

```python
# Bad: Modifying existing class
class EpochPool:
    def forward(self, h_chan, new_param):  # Breaking change!
        ...

# Good: New class with same interface
class AdvancedEpochPool(EpochPool):
    def forward(self, h_chan):  # Same interface
        ...
```

### Debugging Support

All forward passes return a dictionary with intermediate tensors:

```python
output = {
    'logits': [B, 1],
    'intermediates': {
        'H': [B, E, N, T, D],  # Patch tokens
        'A': [B, E, T, N, N],  # Adjacency matrices
        'Z': [B, E, T, N, D],  # GNN outputs
        'u': [B, E, D],        # Epoch embeddings
        'z': [B, D]            # Night embedding
    }
}
```

## Data Format

### Input

PSG data should be formatted as:
- **Shape**: `[B, E, N, L]`
- **B**: Number of subjects (batch size)
- **E**: Number of 30-second epochs
- **N**: Number of channels (10 for standard PSG)
- **L**: Samples per epoch (e.g., 3000 for 100 Hz × 30s)

### Channels

Standard PSG channels (N=10):
1. EEG F3-M2
2. EEG F4-M1
3. EEG C3-M2
4. EEG C4-M1
5. EEG O1-M2
6. EEG O2-M1
7. EOG Left
8. EOG Right
9. EMG Chin
10. ECG

### Labels

- **ADHD labels**: Binary (0 or 1)
- **Sleep stages** (optional): Integer (0-4)
  - 0: Wake
  - 1: N1
  - 2: N2
  - 3: N3
  - 4: REM

### Preprocessing

Recommended preprocessing:
1. Resample to 100 Hz
2. Bandpass filter (0.3-35 Hz for EEG)
3. Artifact rejection
4. Normalize per channel (z-score or robust scaling)

## Performance Tips

### Overfitting Check (Step 1)

Train on a small subset (5-10 subjects) and verify:
- Training accuracy > 80% (model can learn)
- Test accuracy < 90% (no data leakage)

### Numerical Stability (Step 4)

When adding graphs, check for:
- NaN or Inf in forward pass
- Exploding gradients (gradient clipping recommended)
- Degenerate adjacency matrices (all zeros or all ones)

Solutions:
- Use BatchNorm or LayerNorm
- Lower learning rate (1e-4)
- Add gradient clipping
- Regularize adjacency with sparsity loss

### Memory Optimization

For large datasets:
- Reduce batch size
- Use gradient accumulation
- Reduce `d_model` to 64-128
- Use mixed precision training (fp16)

### Hyperparameter Tuning

Priority order:
1. Learning rate (most important)
2. Patch length and stride
3. Number of transformer layers
4. Graph sparsity regularization
5. Dropout rates

## Citation

If you use this code, please cite the original PatchTST paper:

```bibtex
@article{nie2022patchtst,
  title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author={Nie, Yuqi and Nguyen, Nam H and Sinthong, Phanwadee and Kalagnanam, Jayant},
  journal={arXiv preprint arXiv:2211.14730},
  year={2022}
}
```

## License

This project is built on the PatchTST repository and follows the same license.

## Contact

For questions or issues, please open a GitHub issue.
