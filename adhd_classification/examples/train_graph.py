"""
Example training script for graph-based ADHD classification (Step 4).

This script demonstrates:
1. Token-level outputs from PatchTST
2. Dynamic graph learning
3. GNN for channel interaction modeling
4. Visualizing learned adjacency matrices
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from modules.model import ADHDClassifier
from utils.config import ModelConfig
from utils.data import create_dummy_data, PSGDataset, train_test_split, collate_fn


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        X = batch['X'].to(device)
        y = batch['y'].to(device)

        optimizer.zero_grad()

        output = model(X)
        logits = output['logits'].squeeze(-1)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total += len(y)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)

            output = model(X)
            logits = output['logits'].squeeze(-1)

            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += len(y)

    return total_loss / len(dataloader), correct / total


def visualize_adjacency(model, dataloader, device, save_path='adjacency_vis.png'):
    """Visualize learned adjacency matrices."""
    model.eval()

    with torch.no_grad():
        # Get a batch
        batch = next(iter(dataloader))
        X = batch['X'].to(device)

        # Forward with intermediates
        output = model(X, return_intermediates=True)

        if 'A' in output['intermediates']:
            A = output['intermediates']['A']  # [B, E, T, N, N]

            # Average over batch, epochs, and tokens
            A_mean = A.mean(dim=(0, 1, 2)).cpu().numpy()  # [N, N]

            # Plot
            plt.figure(figsize=(10, 8))
            plt.imshow(A_mean, cmap='viridis', aspect='auto')
            plt.colorbar(label='Connection Strength')
            plt.title('Averaged Learned Adjacency Matrix')
            plt.xlabel('Target Channel')
            plt.ylabel('Source Channel')

            # Channel names (example)
            channel_names = [f'Ch{i+1}' for i in range(A_mean.shape[0])]
            plt.xticks(range(len(channel_names)), channel_names, rotation=45)
            plt.yticks(range(len(channel_names)), channel_names)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()

            print(f"   Adjacency visualization saved to: {save_path}")

            # Print strongest connections
            print("\n   Top 5 strongest connections:")
            flat_idx = np.argsort(A_mean.flatten())[::-1]
            for i in range(5):
                idx = flat_idx[i]
                src = idx // A_mean.shape[1]
                tgt = idx % A_mean.shape[1]
                strength = A_mean[src, tgt]
                print(f"   {i+1}. Ch{src+1} → Ch{tgt+1}: {strength:.4f}")


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
    test_dataset = PSGDataset(test_data['X'], test_data['y'])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Step 4: Graph-based model
    print("\n2. Creating graph-based model...")
    config = ModelConfig.step4_graph(seq_len=3000)
    model = ADHDClassifier(**config.to_dict()).to(device)

    print(f"   Model parameters: {model.get_num_params():,}")
    print(f"   Encoder: {config.encoder_type}")
    print(f"   Graph learner: {config.graph_learner_type}")
    print(f"   GNN type: {config.gnn_type}")
    print(f"   GNN layers: {config.gnn_kwargs['n_layers']}")

    # Check numerical stability
    print("\n3. Testing forward pass (numerical stability check)...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        X_sample = sample_batch['X'].to(device)

        try:
            output = model(X_sample, return_intermediates=True)
            print("   ✓ Forward pass successful")

            # Check for NaN or Inf
            has_nan = torch.isnan(output['logits']).any().item()
            has_inf = torch.isinf(output['logits']).any().item()

            if has_nan or has_inf:
                print("   ✗ NaN or Inf detected in outputs!")
            else:
                print("   ✓ No numerical issues detected")

            # Print shapes
            print(f"\n   Output shapes:")
            print(f"   - Logits: {output['logits'].shape}")

            intermediates = output['intermediates']
            for key in ['H', 'A', 'Z', 'u', 'z']:
                if key in intermediates:
                    value = intermediates[key]
                    if isinstance(value, torch.Tensor):
                        print(f"   - {key}: {value.shape}")

        except Exception as e:
            print(f"   ✗ Forward pass failed: {e}")
            return

    # Train
    print("\n4. Training graph model...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epochs = 50
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_step4.pt')

    # Load best model
    model.load_state_dict(torch.load('best_model_step4.pt'))

    # Final evaluation
    print("\n5. Final evaluation...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"   Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Visualize learned graphs
    print("\n6. Visualizing learned adjacency matrices...")
    visualize_adjacency(model, test_loader, device, save_path='adjacency_matrices.png')

    # Compare with non-graph baseline
    print("\n7. Comparing with Step 2 (PatchTST without graph)...")
    config_step2 = ModelConfig.step2_patchtst(seq_len=3000)
    model_step2 = ADHDClassifier(**config_step2.to_dict()).to(device)
    optimizer2 = optim.Adam(model_step2.parameters(), lr=1e-4, weight_decay=1e-5)

    # Train briefly
    for epoch in range(50):
        train_epoch(model_step2, train_loader, optimizer2, criterion, device)

    val_loss_step2, _ = evaluate(model_step2, val_loader, criterion, device)

    print(f"\n8. Results comparison:")
    print(f"   Step 2 (PatchTST only) - Val Loss: {val_loss_step2:.4f}")
    print(f"   Step 4 (With Graph) - Val Loss: {best_val_loss:.4f}")

    if best_val_loss < val_loss_step2:
        print("   ✓ Graph model improves over baseline")
    else:
        print("   ⚠ Graph model does not improve (may need tuning)")

    print("\n✓ Graph-based training complete!")
    print(f"   Model saved to: best_model_step4.pt")


if __name__ == '__main__':
    main()
