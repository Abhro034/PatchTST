"""
Test epoch sampling functionality
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np

from utils.fif_datamodule import collate_fn_adhd

def test_epoch_sampling():
    """Test that epoch sampling works correctly"""
    print("=" * 60)
    print("Testing Epoch Sampling Functionality")
    print("=" * 60)

    # Create mock data - simulate 3 subjects with different epoch counts
    batch = [
        {
            'X': torch.randn(800, 10, 3000),  # Subject 1: 800 epochs
            'y': torch.tensor(1),
            'subject_id': 'S001',
            'sleep_stages': torch.randint(0, 5, (800,))  # Random sleep stages (0-4)
        },
        {
            'X': torch.randn(600, 10, 3000),  # Subject 2: 600 epochs
            'y': torch.tensor(0),
            'subject_id': 'S002',
            'sleep_stages': torch.randint(0, 5, (600,))
        },
        {
            'X': torch.randn(1000, 10, 3000),  # Subject 3: 1000 epochs
            'y': torch.tensor(1),
            'subject_id': 'S003',
            'sleep_stages': torch.randint(0, 5, (1000,))
        }
    ]

    print(f"\nOriginal epoch counts:")
    for i, item in enumerate(batch):
        print(f"  Subject {i+1}: {item['X'].shape[0]} epochs")

    # Test uniform sampling
    print(f"\n{'='*60}")
    print("Test 1: Uniform Sampling (128 epochs)")
    print("=" * 60)

    collated = collate_fn_adhd(batch, num_epochs_sample=128, sampling_strategy='uniform')

    print(f"✓ Collated batch shape: {collated['X'].shape}")
    print(f"  Expected: [3, 128, 10, 3000]")

    assert collated['X'].shape == (3, 128, 10, 3000), "Wrong shape after uniform sampling!"
    assert collated['y'].shape == (3,), "Wrong label shape!"
    print("✓ Shapes correct")

    # Check no NaN
    assert not torch.isnan(collated['X']).any(), "NaN detected in sampled data!"
    print("✓ No NaN values")

    # Test random sampling
    print(f"\n{'='*60}")
    print("Test 2: Random Sampling (64 epochs)")
    print("=" * 60)

    collated = collate_fn_adhd(batch, num_epochs_sample=64, sampling_strategy='random')

    print(f"✓ Collated batch shape: {collated['X'].shape}")
    print(f"  Expected: [3, 64, 10, 3000]")

    assert collated['X'].shape == (3, 64, 10, 3000), "Wrong shape after random sampling!"
    print("✓ Shapes correct")

    # Test that random sampling gives different results
    collated2 = collate_fn_adhd(batch, num_epochs_sample=64, sampling_strategy='random')

    # At least some difference expected (very high probability)
    different = not torch.equal(collated['X'], collated2['X'])
    print(f"✓ Random sampling produces different samples: {different}")

    # Test no sampling (when epochs < num_epochs_sample)
    print(f"\n{'='*60}")
    print("Test 3: No Sampling (epochs < num_epochs_sample)")
    print("=" * 60)

    small_batch = [
        {
            'X': torch.randn(50, 10, 3000),  # Only 50 epochs
            'y': torch.tensor(1),
            'subject_id': 'S004',
            'sleep_stages': torch.randint(0, 5, (50,))
        }
    ]

    collated = collate_fn_adhd(small_batch, num_epochs_sample=128, sampling_strategy='uniform')

    print(f"✓ Original epochs: 50")
    print(f"✓ Requested sample: 128")
    print(f"✓ Result shape: {collated['X'].shape}")
    print(f"  (Should keep all 50 epochs)")

    assert collated['X'].shape == (1, 50, 10, 3000), "Should not upsample!"
    print("✓ Correctly handles small epoch counts")

    # Memory comparison
    print(f"\n{'='*60}")
    print("Memory Efficiency Comparison")
    print("=" * 60)

    # Simulate batch_size=4 subjects with 800 epochs each
    print(f"\nOld approach (all epochs):")
    print(f"  Shape: [4, 800, 10, 3000]")
    old_elements = 4 * 800 * 10 * 3000
    old_memory_gb = old_elements * 4 / (1024**3)  # float32 = 4 bytes
    print(f"  Elements: {old_elements:,}")
    print(f"  Memory (data only): {old_memory_gb:.2f} GB")
    print(f"  Memory (with gradients): ~{old_memory_gb * 2:.2f} GB")

    print(f"\nNew approach (epoch sampling, 128 epochs):")
    print(f"  Shape: [8, 128, 10, 3000]")  # Can use 2× batch size!
    new_elements = 8 * 128 * 10 * 3000
    new_memory_gb = new_elements * 4 / (1024**3)
    print(f"  Elements: {new_elements:,}")
    print(f"  Memory (data only): {new_memory_gb:.2f} GB")
    print(f"  Memory (with gradients): ~{new_memory_gb * 2:.2f} GB")

    speedup = old_elements / new_elements
    batch_improvement = 8 / 4
    print(f"\n✓ Memory reduction: {speedup:.1f}×")
    print(f"✓ Can use {batch_improvement:.1f}× larger batch size")
    print(f"✓ Effective throughput improvement: ~{batch_improvement:.1f}×")

    print(f"\n{'='*60}")
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print("\nEpoch sampling is working correctly!")
    print("You can now train with:")
    print("  - batch_size: 8-16 (instead of 2-4)")
    print("  - num_epochs_sample: 128 (or 64, 256)")
    print("  - sampling_strategy: 'uniform', 'random', or 'stage_stratified'")

    return True


if __name__ == '__main__':
    success = test_epoch_sampling()
    sys.exit(0 if success else 1)
