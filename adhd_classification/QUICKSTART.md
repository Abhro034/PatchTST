# Quick Start Guide

Get started with the ADHD classification pipeline in 5 minutes.

## Installation

```bash
# Clone repository
git clone <repo-url>
cd PatchTST/adhd_classification

# Install dependencies
pip install torch numpy matplotlib scikit-learn

# No additional dependencies needed - uses PyTorch only!
```

## Basic Usage

### Step 1: Create Dummy Data

```python
from utils.data import create_dummy_data, PSGDataset, train_test_split
from torch.utils.data import DataLoader

# Generate dummy data for testing
data = create_dummy_data(
    n_subjects=100,
    n_epochs=200,
    n_channels=10,
    seq_len=3000
)

# Split into train/val/test
train_data, val_data, test_data = train_test_split(data)

# Create datasets
train_dataset = PSGDataset(train_data['X'], train_data['y'])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```

### Step 2: Create Model

```python
from modules.model import ADHDClassifier
from utils.config import ModelConfig

# Use pre-configured model for Step 1 (TCN baseline)
config = ModelConfig.step1_tcn_baseline()
model = ADHDClassifier(**config.to_dict())

# Or use Step 4 (Graph-based)
config = ModelConfig.step4_graph(seq_len=3000)
model = ADHDClassifier(**config.to_dict())
```

### Step 3: Train

```python
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(100):
    for batch in train_loader:
        X = batch['X'].to(device)
        y = batch['y'].to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output['logits'].squeeze(-1), y)
        loss.backward()
        optimizer.step()
```

## Run Examples

The fastest way to get started is to run the example scripts:

```bash
cd examples

# Step 1: TCN baseline (fastest, ~2 min)
python train_baseline.py

# Step 2: PatchTST encoder (~3 min)
python train_patchtst.py

# Step 4: Graph-based model (~5 min)
python train_graph.py
```

## Progressive Development Path

Follow these steps to build your model:

### 1️⃣ Start with TCN Baseline

```python
config = ModelConfig.step1_tcn_baseline()
model = ADHDClassifier(**config.to_dict())
```

**Goals:**
- Verify data pipeline works
- Check for overfitting on small subset
- Establish baseline performance

**Expected:** Training acc > 80%, Test acc < 90%

### 2️⃣ Upgrade to PatchTST

```python
config = ModelConfig.step2_patchtst(seq_len=3000)
model = ADHDClassifier(**config.to_dict())
```

**Goals:**
- Improve temporal encoding
- Verify API compatibility

**Expected:** Match or exceed TCN performance

### 3️⃣ Add Token Outputs

```python
config = ModelConfig.step3_tokens(seq_len=3000)
model = ADHDClassifier(**config.to_dict())
```

**Goals:**
- Expose patch tokens
- Prepare for graph learning

**Expected:** Similar performance to Step 2

### 4️⃣ Add Graph Learning

```python
config = ModelConfig.step4_graph(seq_len=3000)
model = ADHDClassifier(**config.to_dict())
```

**Goals:**
- Model channel interactions
- Check numerical stability

**Expected:** Improved performance (if meaningful interactions exist)

### 5️⃣ Advanced Features

```python
config = ModelConfig.step5_advanced(seq_len=3000)
model = ADHDClassifier(**config.to_dict())
```

**Goals:**
- Top-k pooling
- Multi-task learning
- Confound control

**Expected:** Best performance

## Configuration Cheat Sheet

### Quick Configurations

```python
from utils.config import ModelConfig

# Step 1: TCN baseline
config = ModelConfig.step1_tcn_baseline()

# Step 2: PatchTST
config = ModelConfig.step2_patchtst(seq_len=3000)

# Step 3: Tokens
config = ModelConfig.step3_tokens(seq_len=3000)

# Step 4: Graph
config = ModelConfig.step4_graph(seq_len=3000)

# Step 5: Advanced
config = ModelConfig.step5_advanced(seq_len=3000)

# Use config
model = ADHDClassifier(**config.to_dict())
```

### Custom Configuration

```python
config = ModelConfig(
    n_channels=10,
    seq_len=3000,
    d_model=128,
    encoder_type='patchtst',
    encoder_kwargs={
        'patch_len': 50,
        'stride': 25,
        'n_layers': 4
    },
    use_graph=True,
    gnn_type='gat'
)
```

## Common Issues

### Issue: Forward pass fails with NaN

**Solution:**
```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Lower learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Add regularization
config.graph_learner_kwargs = {'sparsity_reg': 0.01}
```

### Issue: Model doesn't overfit on small subset

**Solution:**
```python
# Increase model capacity
config.d_model = 256
config.encoder_kwargs['n_layers'] = 6

# Remove regularization
config.head_dropout = 0.0
optimizer = optim.Adam(model.parameters(), weight_decay=0.0)
```

### Issue: Out of memory

**Solution:**
```python
# Reduce batch size
batch_size = 4  # or 2

# Reduce model size
config.d_model = 64
config.encoder_kwargs['n_layers'] = 2

# Reduce sequence length (if possible)
# Or use gradient accumulation
```

## Next Steps

1. **Read the full README** for detailed architecture explanation
2. **Explore the modules** in `modules/` to understand each component
3. **Modify configurations** in `utils/config.py` for your use case
4. **Add your own data** by implementing a custom `PSGDataset`
5. **Experiment with hyperparameters** following the priority order in README

## Getting Help

- Check the [README](README.md) for detailed documentation
- Look at example scripts in `examples/`
- Examine module docstrings for API details
- Open an issue for bugs or questions

Happy coding! 🚀
