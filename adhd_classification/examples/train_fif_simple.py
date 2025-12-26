"""
Simple Training Script for ADHD Classification using FIF data

Train/val/test split without cross-validation for faster experimentation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm

from modules.model import ADHDClassifier
from utils.config import ModelConfig
from utils.fif_datamodule import create_dataloaders


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="Training"):
        X = batch['X'].to(device)  # [B, E, N, L]
        y = batch['y'].to(device)  # [B]

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

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            X = batch['X'].to(device)
            y = batch['y'].to(device)

            output = model(X)
            logits = output['logits'].squeeze(-1)

            loss = criterion(logits, y)

            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == y).sum().item()
            total += len(y)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    # AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return avg_loss, accuracy, auc


def main():
    """Main training function"""

    # Configuration
    config = {
        'fif_directory': './data/fif_files',  # UPDATE THIS PATH
        'batch_size': 4,
        'num_workers': 0,
        'step': 4,  # Model step (1-5)
        'num_epochs': 50,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'save_dir': './results/adhd_simple'
    }

    print("=" * 80)
    print("ADHD Classification - Simple Training")
    print("=" * 80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Check FIF directory
    if not Path(config['fif_directory']).exists():
        print(f"\nERROR: FIF directory not found: {config['fif_directory']}")
        print("\nPlease update the 'fif_directory' path in the config")
        return

    # Load data
    print("\nLoading data...")
    try:
        train_loader, val_loader, test_loader, metadata = create_dataloaders(
            fif_directory=config['fif_directory'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
    except Exception as e:
        print(f"\nERROR loading data: {e}")
        return

    print(f"\nDataset info:")
    print(f"  Train subjects: {metadata['n_train']}")
    print(f"  Val subjects: {metadata['n_val']}")
    print(f"  Test subjects: {metadata['n_test']}")
    print(f"  Channels: {metadata['n_channels']}")
    print(f"  Sequence length: {metadata['seq_len']}")
    print(f"  ADHD: {metadata['adhd_count']}, Control: {metadata['control_count']}")

    # Create model
    print("\nCreating model...")
    if config['step'] == 1:
        model_config = ModelConfig.step1_tcn_baseline()
    elif config['step'] == 2:
        model_config = ModelConfig.step2_patchtst(seq_len=metadata['seq_len'])
    elif config['step'] == 3:
        model_config = ModelConfig.step3_tokens(seq_len=metadata['seq_len'])
    elif config['step'] == 4:
        model_config = ModelConfig.step4_graph(seq_len=metadata['seq_len'])
    elif config['step'] == 5:
        model_config = ModelConfig.step5_advanced(seq_len=metadata['seq_len'])
    else:
        print(f"ERROR: Unknown step {config['step']}")
        return

    model = ADHDClassifier(**model_config.to_dict()).to(device)
    print(f"  Model parameters: {model.get_num_params():,}")
    print(f"  Using Step {config['step']}: {model_config.encoder_type}")

    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nTraining for {config['num_epochs']} epochs...")
    best_val_auc = 0.0
    best_epoch = 0

    for epoch in range(config['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1:3d}/{config['num_epochs']} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
            print(f"  → New best model (AUC: {val_auc:.4f})")

    # Load best model and evaluate on test set
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(save_dir / 'best_model.pt'))

    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Best Val AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC: {test_auc:.4f}")

    print(f"\nModel saved to: {save_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
