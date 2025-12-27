# Epoch Sampling Strategy for Memory-Efficient Training

## Problem

Subject-level ADHD classification requires processing all epochs from each subject together, but this creates memory issues:

- **Subject-wise batching**: Each subject has 600-1000 epochs
- **Memory per batch**: batch_size × epochs × channels × samples
  - Example: 4 subjects × 800 epochs × 10 channels × 3000 samples = **96M values** → ~6-12 GB with gradients
- **Result**: Can only use tiny batch sizes (1-2) on 24GB GPU, leading to slow training and unstable gradients

## Solution: Epoch Sampling with Subject-Level Labels

**Key Insight**: We don't need to process ALL epochs in one forward pass. We can:
1. **Sample** a subset of epochs per subject per batch (e.g., 128 epochs)
2. **Train** with subject-level labels (not pretending epochs have labels)
3. **Rotate** through different epochs over training so model sees whole night
4. **Evaluate** by aggregating over all epochs or multiple samplings

## Implementation

###sampling Strategies

#### 1. Uniform Sampling (Default)
Samples epochs evenly spaced across the night:

```python
# For 800 epochs, sample 128 uniformly
step = 800 / 128  # = 6.25
indices = [0, 6, 12, 19, 25, 31, ...]  # Every ~6 epochs
```

**Pros:**
- Covers entire night duration
- Maintains temporal distribution
- Good for capturing sleep progression

#### 2. Random Sampling
Randomly selects epochs each batch:

```python
# Random 128 epochs
indices = random.permute(800)[:128]
```

**Pros:**
- Maximum diversity over training
- Different samples each epoch
- Prevents overfitting to specific epochs

#### 3. Stage-Stratified Sampling
Samples proportionally from each sleep stage:

```python
# If night has: 100 Wake, 200 N1, 300 N2, 150 N3, 50 REM
# Sample proportionally: 16 Wake, 32 N1, 48 N2, 24 N3, 8 REM
```

**Pros:**
- Ensures all sleep stages represented
- Prevents bias toward common stages (N2)
- Important for clinical interpretation

## Memory Comparison

| Approach | Batch Size | Epochs/Subject | Memory | GPU Util |
|----------|------------|----------------|--------|----------|
| **Old: All epochs** | 2 | 800 | ~12 GB | Low |
| **New: Sampled** | 8 | 128 | ~6 GB | High |
| **New: Aggressive** | 16 | 128 | ~12 GB | Very High |

## Usage

### Training Script Configuration

```python
config = {
    'batch_size': 8,              # Now we can use 8 instead of 2!
    'num_epochs_sample': 128,     # Sample 128 epochs per subject
    'sampling_strategy': 'uniform',  # or 'random' or 'stage_stratified'
    ...
}

train_loader, val_loader, test_loader, metadata = create_dataloaders(
    fif_directory='./data',
    batch_size=config['batch_size'],
    num_epochs_sample=config['num_epochs_sample'],
    sampling_strategy=config['sampling_strategy']
)
```

### Evaluation: Full Aggregation

At evaluation time, process **all epochs** to get best performance:

```python
from utils.fif_datamodule import evaluate_subject_full

# Process all epochs in chunks
pred, prob = evaluate_subject_full(
    model,
    subject_data,  # All epochs
    device,
    num_epochs_per_batch=128  # Process 128 at a time
)
```

### Evaluation: Multi-sampling

Alternative: Average over multiple random samplings:

```python
from utils.fif_datamodule import evaluate_subject_multisampling

# Average 5 random samplings
pred, prob = evaluate_subject_multisampling(
    model,
    subject_data,
    device,
    num_samples=5,
    num_epochs_per_sample=128
)
```

## Recommended Settings

### By GPU Memory

| GPU Memory | Batch Size | Epochs/Subject | Sampling Strategy |
|------------|------------|----------------|-------------------|
| 8 GB | 4 | 64 | uniform |
| 12 GB | 4 | 128 | uniform |
| 16 GB | 8 | 128 | uniform |
| **24 GB** | **8-16** | **128-256** | **stage_stratified** |
| 32 GB+ | 16-32 | 256 | stage_stratified |

### By Model Step

| Step | Model | Recommended | Max Possible |
|------|-------|-------------|--------------|
| 1 | TCN | batch=16, epochs=256 | batch=32, epochs=256 |
| 2 | PatchTST | batch=12, epochs=256 | batch=16, epochs=256 |
| 3 | Tokens | batch=8, epochs=128 | batch=12, epochs=256 |
| 4 | Graph | batch=8, epochs=128 | batch=8, epochs=256 |

## Expected Performance

### Speed Improvement

With RTX 4090 (24GB):

| Configuration | Batch Throughput | Epochs/sec | Relative Speed |
|---------------|------------------|------------|----------------|
| Old (batch=2, all epochs) | 2 subj/batch | ~0.5 | 1× (baseline) |
| New (batch=8, sample=128) | 8 subj/batch | ~2.5 | **5×** faster |
| New (batch=16, sample=128) | 16 subj/batch | ~4.0 | **8×** faster |

### Model Performance

The model performance should be **similar or better** because:
1. Still sees all data over training (via sampling rotation)
2. Larger effective batch size → more stable gradients
3. Can train for more epochs due to faster speed
4. Evaluation uses all epochs → no information loss

## Best Practices

1. **Training**: Use `sampling_strategy='random'` for maximum diversity
2. **Validation**: Use `evaluate_subject_full()` to process all epochs
3. **Testing**: Use `evaluate_subject_full()` for final metrics
4. **Experiments**: Try different `num_epochs_sample` values (64, 128, 256)

## Example: Complete Training Loop

```python
# Training
for epoch in range(num_epochs):
    for batch in train_loader:
        # batch['X']: [8, 128, 10, 3000]  # 8 subjects, 128 sampled epochs each
        # Different 128 epochs sampled each iteration!

        X = batch['X'].to(device)
        y = batch['y'].to(device)

        output = model(X)
        loss = criterion(output['logits'].squeeze(-1), y)
        loss.backward()
        optimizer.step()

# Evaluation (all epochs)
all_preds = []
all_labels = []

for subject_data in test_dataset:
    pred, prob = evaluate_subject_full(
        model, subject_data, device
    )
    all_preds.append(pred)
    all_labels.append(subject_data['y'].item())

accuracy = accuracy_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
```

## Theoretical Justification

This approach is valid because:

1. **MIL Framework**: We're treating each subject as a "bag" of epochs
2. **Bag Sampling**: Common in MIL to sample from large bags during training
3. **Full Aggregation**: At test time, we aggregate over the full bag
4. **Stochastic Training**: Different samples per batch provide regularization
5. **Proven**: Used in similar medical time-series classification papers

## References

This strategy is inspired by:
- Multiple Instance Learning with bag sampling
- Stochastic training with data augmentation
- Medical time-series classification best practices
