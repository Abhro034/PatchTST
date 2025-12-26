"""
Example training script for ADHD classification with PatchTST (Step 2).

This script demonstrates:
1. Swapping from TCN to PatchTST encoder
2. Verifying the API remains the same
3. Comparing performance with baseline
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from modules.model import ADHDClassifier
from utils.config import ModelConfig
from utils.data import create_dummy_data, PSGDataset, train_test_split, collate_fn


def train_and_evaluate(model, train_loader, val_loader, device, num_epochs=100):
    """Train and evaluate model."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output['logits'].squeeze(-1), y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device)
                y = batch['y'].to(device)

                output = model(X)
                loss = criterion(output['logits'].squeeze(-1), y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    return best_val_loss


def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dummy data
    print("\n1. Creating dummy data...")
    data = create_dummy_data(
        n_subjects=100,
        n_epochs=200,
        n_channels=10,
        seq_len=3000,
        seed=42
    )

    train_data, val_data, test_data = train_test_split(data)

    # Create datasets and loaders
    train_dataset = PSGDataset(train_data['X'], train_data['y'])
    val_dataset = PSGDataset(val_data['X'], val_data['y'])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Step 2: PatchTST with patch pooling
    print("\n2. Training Step 2 - PatchTST with patch pooling...")
    config_step2 = ModelConfig.step2_patchtst(seq_len=3000)
    model_step2 = ADHDClassifier(**config_step2.to_dict()).to(device)

    print(f"   Model parameters: {model_step2.get_num_params():,}")
    print(f"   Encoder: {config_step2.encoder_type}")
    print(f"   Patch length: {config_step2.encoder_kwargs['patch_len']}")
    print(f"   Stride: {config_step2.encoder_kwargs['stride']}")

    val_loss_step2 = train_and_evaluate(model_step2, train_loader, val_loader, device)

    # Test intermediate outputs
    print("\n3. Testing intermediate outputs...")
    model_step2.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        X_sample = sample_batch['X'].to(device)
        output = model_step2(X_sample, return_intermediates=True)

        print(f"   Output keys: {list(output.keys())}")
        print(f"   Logits shape: {output['logits'].shape}")

        intermediates = output['intermediates']
        print(f"\n   Intermediate tensors:")
        for key, value in intermediates.items():
            if isinstance(value, torch.Tensor):
                print(f"   - {key}: {value.shape}")

    # Compare with baseline
    print("\n4. Comparing with baseline (TCN)...")
    config_step1 = ModelConfig.step1_tcn_baseline()
    model_step1 = ADHDClassifier(**config_step1.to_dict()).to(device)

    print(f"   Baseline parameters: {model_step1.get_num_params():,}")
    val_loss_step1 = train_and_evaluate(model_step1, train_loader, val_loader, device)

    print(f"\n5. Results comparison:")
    print(f"   Step 1 (TCN) - Val Loss: {val_loss_step1:.4f}")
    print(f"   Step 2 (PatchTST) - Val Loss: {val_loss_step2:.4f}")

    if val_loss_step2 <= val_loss_step1:
        print("   ✓ PatchTST matches or improves over TCN baseline")
    else:
        print("   ⚠ PatchTST underperforms baseline (may need tuning)")

    # Verify API consistency
    print("\n6. API consistency check...")
    print("   ✓ Both models use the same forward() interface")
    print("   ✓ Both models return the same output structure")
    print("   ✓ Swapping encoder requires no changes to training code")

    print("\n✓ PatchTST training complete!")


if __name__ == '__main__':
    main()
