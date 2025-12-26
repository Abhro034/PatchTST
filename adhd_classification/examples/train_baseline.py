"""
Example training script for ADHD classification baseline (Step 1).

This script demonstrates:
1. Creating dummy data
2. Building the baseline TCN model
3. Training on a small subset to ensure overfitting
4. Checking for data leakage
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
from utils.config import ModelConfig, TrainingConfig
from utils.data import create_dummy_data, PSGDataset, train_test_split, collate_fn


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        X = batch['X'].to(device)  # [B, E, N, L]
        y = batch['y'].to(device)  # [B]

        optimizer.zero_grad()

        # Forward pass
        output = model(X, return_intermediates=False)
        logits = output['logits'].squeeze(-1)  # [B]

        # Compute loss
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total += len(y)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)

            output = model(X, return_intermediates=False)
            logits = output['logits'].squeeze(-1)

            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += len(y)

            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy, torch.cat(all_logits), torch.cat(all_labels)


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
        adhd_prevalence=0.3,
        seed=42
    )
    print(f"   Data shape: {data['X'].shape}")
    print(f"   ADHD prevalence: {data['y'].mean():.2%}")

    # Split data
    train_data, val_data, test_data = train_test_split(data, test_size=0.2, val_size=0.1)
    print(f"   Train: {len(train_data['y'])} subjects")
    print(f"   Val: {len(val_data['y'])} subjects")
    print(f"   Test: {len(test_data['y'])} subjects")

    # Create datasets
    train_dataset = PSGDataset(train_data['X'], train_data['y'])
    val_dataset = PSGDataset(val_data['X'], val_data['y'])
    test_dataset = PSGDataset(test_data['X'], test_data['y'])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )

    # Create model
    print("\n2. Creating model...")
    config = ModelConfig.step1_tcn_baseline()
    model = ADHDClassifier(**config.to_dict())
    model = model.to(device)

    print(f"   Model parameters: {model.get_num_params():,}")
    print(f"   Configuration: Step {config.step} - {config.encoder_type.upper()}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training
    print("\n3. Training baseline model...")
    print("   (Training on small subset to check overfitting capability)")

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model_step1.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n   Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model_step1.pt'))

    # Final evaluation
    print("\n4. Final evaluation...")
    train_loss, train_acc, train_logits, train_labels = evaluate(model, train_loader, criterion, device)
    val_loss, val_acc, val_logits, val_labels = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc, test_logits, test_labels = evaluate(model, test_loader, criterion, device)

    print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    print(f"   Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # Check for overfitting (should overfit on small subset)
    print("\n5. Overfitting check...")
    if train_acc > 0.8:
        print("   ✓ Model can overfit on training data (good!)")
    else:
        print("   ✗ Model unable to overfit - may have issues")

    # Check for data leakage
    print("\n6. Data leakage check...")
    if test_acc < 0.9:  # Should not be perfect on test set
        print("   ✓ No obvious data leakage detected")
    else:
        print("   ⚠ Test accuracy suspiciously high - check for leakage")

    print("\n7. Testing intermediate outputs...")
    # Test with return_intermediates=True
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        X_sample = sample_batch['X'].to(device)
        output = model(X_sample, return_intermediates=True)

        print(f"   Logits shape: {output['logits'].shape}")
        print(f"   Intermediate keys: {list(output['intermediates'].keys())}")

        for key, value in output['intermediates'].items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")

    print("\n✓ Baseline training complete!")
    print(f"   Model saved to: best_model_step1.pt")


if __name__ == '__main__':
    main()
