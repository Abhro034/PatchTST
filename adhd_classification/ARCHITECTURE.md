# Architecture Documentation

## System Overview

The ADHD classification pipeline is designed as a modular, swappable system with five progressive implementation steps.

## Core Tensor Shapes

Throughout the pipeline, we maintain these canonical shapes:

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| B | Batch size (subjects) | 8-32 |
| E | Epochs per subject | 150-300 |
| N | Number of channels | 10 |
| L | Samples per epoch | 3000 (100Hz × 30s) |
| D | Embedding dimension | 128-256 |
| T | Patch tokens per epoch | ~60 (depends on patch_len/stride) |

## Module-by-Module Flow

### 1. Input Processing

```
Input: [B, E, N, L]
Example: [8, 200, 10, 3000]
```

Each subject has E epochs, each epoch has N channels, each channel has L samples.

### 2. Temporal Encoder

**Reshape:**
```
[B, E, N, L] → [B*E*N, L]
[8, 200, 10, 3000] → [16000, 3000]
```

**Process per channel:**

#### Option A: TCN (Step 1)
```
TemporalConvNet: [B*E*N, 1, L] → [B*E*N, D]
                 [16000, 1, 3000] → [16000, 128]
```

#### Option B: PatchTST (Step 2)
```
Patching: [B*E*N, 1, L] → [B*E*N, 1, T, patch_len]
          [16000, 1, 3000] → [16000, 1, 60, 50]

Transformer: [B*E*N, 1, T, D]
             [16000, 1, 60, 128]

Pool patches: [B*E*N, D]
              [16000, 128]
```

#### Option C: PatchTST with tokens (Step 3+)
```
Transformer: [B*E*N, 1, T, D]
             [16000, 1, 60, 128]

Keep tokens: [B*E*N, T, D]
             [16000, 60, 128]
```

**Reshape back:**
```
[B*E*N, D] → [B, E, N, D]
[16000, 128] → [8, 200, 10, 128]

Or with tokens:
[B*E*N, T, D] → [B, E, N, T, D]
[16000, 60, 128] → [8, 200, 10, 60, 128]
```

### 3. Epoch Pooling

**Without graph (Steps 1-3):**
```
Input: [B, E, N, D]
       [8, 200, 10, 128]

Mean/Attention pool across N:
Output: [B, E, D]
        [8, 200, 128]
```

**With graph (Steps 4-5):**
```
Input: [B, E, N, T, D]
       [8, 200, 10, 60, 128]

For each (epoch e, token t):
  Node features: [B, N, D]
                 [8, 10, 128]

  GraphLearner: [B, N, D] → [B, N, N]
                [8, 10, 128] → [8, 10, 10]

  GNNBlock: [B, N, D], [B, N, N] → [B, N, D]
            [8, 10, 128], [8, 10, 10] → [8, 10, 128]

Pool over N nodes: [B, T, D]
                   [8, 60, 128]

Pool over T tokens: [B, D]
                    [8, 128]

Across all E epochs:
Output: [B, E, D]
        [8, 200, 128]
```

### 4. Night Pooling

**Simple pooling:**
```
Input: [B, E, D]
       [8, 200, 128]

Mean/Max/Attention across E:
Output: [B, D]
        [8, 128]
```

**Top-k pooling:**
```
Input: [B, E, D]
       [8, 200, 128]

Compute attention scores:
Scores: [B, E]
        [8, 200]

Select top k=20 epochs:
TopK: [B, k, D]
      [8, 20, 128]

Weighted sum:
Output: [B, D]
        [8, 128]
```

**Stage-aware pooling:**
```
Input: [B, E, D], stage_labels: [B, E]
       [8, 200, 128], [8, 200]

For each sleep stage s ∈ {0,1,2,3,4}:
  Mask epochs with stage s
  Pool within stage: [B, D]
                     [8, 128]

Concatenate: [B, 5*D]
             [8, 640]

Project: [B, D]
         [8, 128]
```

### 5. Classification Head

```
Input: [B, D]
       [8, 128]

Linear → ReLU → Dropout → Linear:
Output: [B, 1]
        [8, 1]
```

## Graph Construction Details (Step 4+)

### Token Binning Strategy

**Recommended: 5-second bins (~6 graphs per epoch)**

```
30-second epoch → 6 bins of 5 seconds each

With patch_len=50 (0.5s) and stride=25:
  - Total patches per epoch: ~60
  - Patches per 5s bin: ~10

Graph construction:
  For each 5s bin:
    Node features: [B, N, 10, D]
                   [8, 10, 10, 128]

    Pool across 10 tokens:
    X_nodes: [B, N, D]
             [8, 10, 128]

    Learn graph:
    A: [B, N, N]
       [8, 10, 10]

    Apply GNN:
    Z: [B, N, D]
       [8, 10, 128]

Total per epoch: 6 graphs → [B, 6, N, D]
                            [8, 6, 10, 128]

Pool over 6 bins + N nodes:
Epoch embedding: [B, D]
                 [8, 128]
```

### Graph Learner: Attention-based

```python
def forward(x):
    # x: [B, N, D]

    Q = W_q(x)  # [B, N, D]
    K = W_k(x)  # [B, N, D]

    scores = (Q @ K^T) / sqrt(D)  # [B, N, N]
    A = softmax(scores, dim=-1)   # [B, N, N]

    return A
```

### GNN Block: GAT

```python
def forward(x, A):
    # x: [B, N, D]
    # A: [B, N, N]

    h = W(x)  # [B, N, D']

    # Multi-head attention
    for each head h:
        # Compute attention coefficients
        e_ij = LeakyReLU(a^T [h_i || h_j])

        # Use A as mask
        e_ij = e_ij * A

        # Normalize
        alpha_ij = softmax(e_ij)

        # Aggregate
        h'_i = Σ_j alpha_ij * h_j

    return h'
```

## Multi-task Learning (Step 5)

### Architecture

```
                    ┌─────────────┐
                    │   Encoder   │
                    └──────┬──────┘
                           │
                    [B, E, D]
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │  Sleep   │  │ Arousal  │  │   ADHD   │
       │  Stage   │  │Detection │  │Classifier│
       │   Head   │  │   Head   │  │   Head   │
       └──────────┘  └──────────┘  └──────────┘
           │              │              │
       [B,E,5]        [B,E,1]        [B,1]
```

### Loss Computation

```python
# Primary task
adhd_loss = BCE(adhd_logits, adhd_labels)

# Auxiliary tasks
stage_loss = CE(stage_logits, stage_labels)
arousal_loss = BCE(arousal_logits, arousal_labels)

# Combined
total_loss = adhd_loss + 0.3 * (stage_loss + arousal_loss)
```

## Confound Control (Step 5)

### Adversarial Learning

```
                    ┌─────────────┐
                    │   Encoder   │
                    └──────┬──────┘
                           │
                        [B, D]
                           │
              ┌────────────┼────────────┐
              │                         │
              ▼                         ▼
       ┌──────────┐              ┌──────────┐
       │   ADHD   │              │ Confound │
       │Classifier│              │Predictors│
       │(maximize)│              │(minimize)│
       └──────────┘              └──────────┘
```

### Training Strategy

```python
# Forward pass
z = encoder(x)  # [B, D]
adhd_logits = classifier(z)
confound_preds = {k: head(z) for k, head in confound_heads.items()}

# Losses
adhd_loss = BCE(adhd_logits, adhd_labels)
confound_loss = Σ MSE(confound_preds[k], confounds[k])

# Encoder: minimize ADHD loss, maximize confound loss (via gradient reversal)
# Classifier: minimize ADHD loss
# Confound heads: minimize confound loss
```

## Memory Requirements

### Step 1 (TCN)

```
Parameters: ~500K
Peak memory (batch=8, E=200):
  - Input: 8×200×10×3000×4 = 192 MB
  - Embeddings: 8×200×10×128×4 = 6.5 MB
  - Activations: ~50 MB
Total: ~250 MB
```

### Step 4 (Graph)

```
Parameters: ~2M
Peak memory (batch=8, E=200):
  - Input: 192 MB
  - Tokens: 8×200×10×60×128×4 = 390 MB
  - Adjacency: 8×200×6×10×10×4 = 3.8 MB (6 graphs/epoch)
  - Activations: ~150 MB
Total: ~750 MB
```

## Computational Complexity

### TemporalEncoder (PatchTST)

```
Input: [B*E*N, T, D]

Attention: O(T^2 * D) per layer
With T=60, D=128, 4 layers:
  ~60^2 * 128 * 4 = 1.8M operations per channel

For B=8, E=200, N=10:
  Total: 28.8B operations
```

### GraphLearner

```
Input: [B*E*T, N, D]

Attention: O(N^2 * D)
With N=10, D=128:
  ~10^2 * 128 = 13K operations per graph

For B=8, E=200, T=6:
  Total: 120M operations
```

### GNNBlock (2 layers)

```
Input: [B*E*T, N, D]

Per layer: O(N^2 * D)
With N=10, D=128, 2 layers:
  ~10^2 * 128 * 2 = 26K operations per graph

For B=8, E=200, T=6:
  Total: 250M operations
```

## Design Patterns

### 1. Swappable Components

```python
class Component(nn.Module):
    def __init__(self, **kwargs):
        # Parse kwargs for flexibility
        pass

    def forward(self, x):
        # Fixed input/output interface
        return output
```

### 2. Configuration-driven

```python
config = {
    'encoder_type': 'patchtst',
    'pool_type': 'attention',
    'use_graph': True
}

model = build_from_config(config)
```

### 3. Debug-friendly

```python
output = model(x, return_intermediates=True)

# Access any intermediate tensor
H = output['intermediates']['H']
A = output['intermediates']['A']
```

### 4. Single Responsibility

Each module does ONE thing:
- TemporalEncoder: Encode time-series
- EpochPool: Fuse channels
- NightPool: Aggregate epochs
- GraphLearner: Learn adjacency
- GNNBlock: Message passing

## Extension Points

### Adding New Encoder

```python
class MyEncoder(nn.Module):
    def forward(self, x):
        # x: [B*E*N, L]
        # return: [B*E*N, D] or [B*E*N, T, D]
        return output

# Use in TemporalEncoder
encoder = TemporalEncoder(
    encoder_type='custom',
    custom_encoder_class=MyEncoder
)
```

### Adding New Pooling

```python
class MyPooling(nn.Module):
    def forward(self, x):
        # x: [B, E, N, D] or [B, E, N, T, D]
        # return: [B, E, D]
        return output

# Use in EpochPool
pool = EpochPool(pool_type='custom', custom_pool_class=MyPooling)
```

### Adding New GNN

```python
class MyGNN(nn.Module):
    def forward(self, x, A):
        # x: [B, N, D]
        # A: [B, N, N]
        # return: [B, N, D]
        return output

# Use in GNNBlock
gnn = GNNBlock(gnn_type='custom', custom_gnn_class=MyGNN)
```

## Testing Strategy

### Unit Tests

Test each module independently:

```python
def test_temporal_encoder():
    x = torch.randn(100, 3000)  # [B*E*N, L]
    encoder = TemporalEncoder(seq_len=3000, d_model=128)
    out = encoder(x)
    assert out.shape == (100, 128)

def test_graph_learner():
    x = torch.randn(8, 10, 128)  # [B, N, D]
    learner = GraphLearner(d_model=128)
    A = learner(x)
    assert A.shape == (8, 10, 10)
    assert torch.allclose(A.sum(dim=-1), torch.ones(8, 10))  # Row-normalized
```

### Integration Tests

Test full pipeline:

```python
def test_step4_pipeline():
    x = torch.randn(8, 200, 10, 3000)
    model = ADHDClassifier(**ModelConfig.step4_graph().to_dict())
    output = model(x, return_intermediates=True)

    assert output['logits'].shape == (8, 1)
    assert 'H' in output['intermediates']
    assert 'A' in output['intermediates']
    assert 'Z' in output['intermediates']
```

### Overfitting Test

```python
def test_overfitting():
    # Small dataset
    data = create_dummy_data(n_subjects=10)
    dataset = PSGDataset(data['X'], data['y'])
    loader = DataLoader(dataset, batch_size=5)

    model = ADHDClassifier(...)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(100):
        train_epoch(model, loader, optimizer)

    # Evaluate
    acc = evaluate(model, loader)

    # Should overfit
    assert acc > 0.9
```
