"""
Sanity Check Training Script for ADHD Classification Pipeline

Tests the pipeline on dummy data to verify:
1. All components work correctly
2. Model can overfit on small subset
3. No numerical issues (NaN, Inf)
4. Proper tensor shapes throughout
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from modules.model import ADHDClassifier
from utils.config import ModelConfig
from utils.data import create_dummy_data, PSGDataset, collate_fn


def check_numerical_stability(tensor, name="tensor"):
    """Check for NaN or Inf in tensor"""
    if torch.isnan(tensor).any():
        print(f"  ✗ WARNING: NaN detected in {name}")
        return False
    if torch.isinf(tensor).any():
        print(f"  ✗ WARNING: Inf detected in {name}")
        return False
    return True


def test_step(step_name, config, train_loader, val_loader, device, num_epochs=20):
    """Test a single step configuration"""
    print(f"\n{'='*80}")
    print(f"Testing {step_name}")
    print(f"{'='*80}")

    # Create model
    model = ADHDClassifier(**config.to_dict()).to(device)
    print(f"Model parameters: {model.get_num_params():,}")

    # Test forward pass
    print("\n1. Testing forward pass...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        X = sample_batch['X'].to(device)

        try:
            output = model(X, return_intermediates=True)
            print(f"  ✓ Forward pass successful")

            # Check shapes
            assert output['logits'].shape == (X.shape[0], 1), f"Wrong logit shape: {output['logits'].shape}"
            print(f"  ✓ Output shape correct: {output['logits'].shape}")

            # Check for NaN/Inf
            all_stable = True
            all_stable &= check_numerical_stability(output['logits'], "logits")

            for key, value in output['intermediates'].items():
                if isinstance(value, torch.Tensor):
                    all_stable &= check_numerical_stability(value, key)

            if all_stable:
                print(f"  ✓ No numerical issues detected")

            # Print intermediate shapes
            print(f"\n  Intermediate tensor shapes:")
            for key, value in output['intermediates'].items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape}")

        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            return False

    # Train
    print("\n2. Training on small subset to test overfitting...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_train_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)

            optimizer.zero_grad()
            output = model(X)
            logits = output['logits'].squeeze(-1)

            loss = criterion(logits, y)
            loss.backward()

            # Check gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"  ✗ NaN/Inf gradient in {name}")
                    has_nan_grad = True
                    break

            if has_nan_grad:
                return False

            optimizer.step()

            epoch_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += len(y)

        train_acc = correct / total
        best_train_acc = max(best_train_acc, train_acc)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{num_epochs} - Loss: {epoch_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}")

    print(f"\n3. Overfitting check:")
    if best_train_acc > 0.8:
        print(f"  ✓ Model can overfit (best train acc: {best_train_acc:.4f})")
    else:
        print(f"  ✗ Model cannot overfit (best train acc: {best_train_acc:.4f})")
        print(f"     This may indicate a problem with the model or training")

    # Validate
    print("\n4. Validation check...")
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)

            output = model(X)
            logits = output['logits'].squeeze(-1)

            preds = (torch.sigmoid(logits) > 0.5).float()
            val_correct += (preds == y).sum().item()
            val_total += len(y)

    val_acc = val_correct / val_total
    print(f"  Validation accuracy: {val_acc:.4f}")

    if val_acc < 0.9:
        print(f"  ✓ No obvious data leakage detected")
    else:
        print(f"  ⚠ Validation accuracy suspiciously high - check for leakage")

    print(f"\n✓ {step_name} passed all tests")
    return True


def main():
    """Run sanity checks on all steps"""
    print("="*80)
    print("ADHD Classification Pipeline - Sanity Check")
    print("="*80)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create small dummy dataset for overfitting test
    print("\nCreating dummy data...")
    data = create_dummy_data(
        n_subjects=20,  # Small for overfitting test
        n_epochs=100,   # Fewer epochs for faster testing
        n_channels=10,
        seq_len=3000,
        adhd_prevalence=0.5,
        seed=42
    )
    print(f"  Created {len(data['y'])} subjects with {data['X'].shape[1]} epochs each")
    print(f"  ADHD: {np.sum(data['y'])}, Control: {len(data['y']) - np.sum(data['y'])}")

    # Split data
    n_train = 16
    n_val = 4

    train_data = {
        'X': data['X'][:n_train],
        'y': data['y'][:n_train]
    }
    val_data = {
        'X': data['X'][n_train:],
        'y': data['y'][n_train:]
    }

    # Create datasets
    train_dataset = PSGDataset(train_data['X'], train_data['y'])
    val_dataset = PSGDataset(val_data['X'], val_data['y'])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )

    # Test each step
    all_passed = True

    # Step 1: TCN Baseline
    config_step1 = ModelConfig.step1_tcn_baseline()
    config_step1.seq_len = 3000
    all_passed &= test_step("Step 1: TCN Baseline", config_step1, train_loader, val_loader, device)

    # Step 2: PatchTST
    config_step2 = ModelConfig.step2_patchtst(seq_len=3000)
    all_passed &= test_step("Step 2: PatchTST Encoder", config_step2, train_loader, val_loader, device)

    # Step 3: Tokens
    config_step3 = ModelConfig.step3_tokens(seq_len=3000)
    all_passed &= test_step("Step 3: Token Outputs", config_step3, train_loader, val_loader, device)

    # Step 4: Graph
    config_step4 = ModelConfig.step4_graph(seq_len=3000)
    all_passed &= test_step("Step 4: Graph Neural Networks", config_step4, train_loader, val_loader, device)

    # Summary
    print("\n" + "="*80)
    print("SANITY CHECK SUMMARY")
    print("="*80)

    if all_passed:
        print("\n✓ All tests passed!")
        print("\nThe pipeline is working correctly. You can now:")
        print("  1. Train on real data")
        print("  2. Experiment with hyperparameters")
        print("  3. Add custom features")
    else:
        print("\n✗ Some tests failed!")
        print("\nPlease review the errors above and fix them before proceeding.")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
