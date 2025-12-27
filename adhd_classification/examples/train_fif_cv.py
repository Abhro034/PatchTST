"""
5-Fold Stratified Cross-Validation for ADHD Classification using FIF data

Subject-level classification with comprehensive metrics
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix,
    classification_report
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import pandas as pd

from modules.model import ADHDClassifier
from utils.config import ModelConfig
from utils.fif_datamodule import load_fif_data, FIFDatasetADHD, collate_fn_adhd


class CrossValidationTrainerADHD:
    """K-Fold Cross-Validation Trainer for ADHD Classification"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Results storage
        self.fold_results = []

        # Create save directory
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def create_model(self, n_channels, seq_len):
        """Create model instance"""
        # Use config to create model
        if self.config['step'] == 1:
            model_config = ModelConfig.step1_tcn_baseline()
        elif self.config['step'] == 2:
            model_config = ModelConfig.step2_patchtst(seq_len=seq_len)
        elif self.config['step'] == 3:
            model_config = ModelConfig.step3_tokens(seq_len=seq_len)
        elif self.config['step'] == 4:
            model_config = ModelConfig.step4_graph(seq_len=seq_len)
        elif self.config['step'] == 5:
            model_config = ModelConfig.step5_advanced(seq_len=seq_len)
        else:
            raise ValueError(f"Unknown step: {self.config['step']}")

        # Override with custom config if provided
        if 'model_config' in self.config:
            for key, value in self.config['model_config'].items():
                setattr(model_config, key, value)

        model = ADHDClassifier(**model_config.to_dict())
        return model.to(self.device)

    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train one epoch"""
        model.train()
        epoch_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            X = batch['X'].to(self.device)  # [B, E, N, L]
            y = batch['y'].to(self.device)  # [B]

            optimizer.zero_grad()

            # Forward
            output = model(X)
            logits = output['logits'].squeeze(-1)  # [B]

            loss = criterion(logits, y)
            loss.backward()

            # Gradient clipping
            if self.config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip'])

            optimizer.step()

            # Track
            epoch_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().detach().numpy()
            preds = (probs > 0.5).astype(int)
            labels = y.cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0

        return avg_loss, acc, auc

    @torch.no_grad()
    def validate(self, model, val_loader, criterion):
        """Validate"""
        model.eval()
        epoch_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        all_subject_ids = []

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            X = batch['X'].to(self.device)
            y = batch['y'].to(self.device)

            output = model(X)
            logits = output['logits'].squeeze(-1)

            loss = criterion(logits, y)

            epoch_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels = y.cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_subject_ids.extend(batch['subject_ids'])

        avg_loss = epoch_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

        # AUC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0

        # Per-class metrics
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

        cm = confusion_matrix(all_labels, all_preds)

        return {
            'loss': avg_loss,
            'accuracy': acc,
            'auc': auc,
            'f1': f1,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels,
            'subject_ids': all_subject_ids
        }

    def train_fold(self, fold, train_dataset, val_dataset):
        """Train one fold"""
        print(f"\n{'='*80}")
        print(f"FOLD {fold + 1}/{self.config['n_folds']}")
        print(f"{'='*80}")

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            collate_fn=collate_fn_adhd,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=collate_fn_adhd,
            pin_memory=True
        )

        # Model
        model = self.create_model(train_dataset.n_channels, train_dataset.n_samples)
        print(f"Model parameters: {model.get_num_params():,}")

        # Optimizer & Scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_epochs'])
        criterion = nn.BCEWithLogitsLoss()

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_auc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'val_f1': []
        }

        best_val_auc = 0.0
        best_epoch = 0

        # Training loop
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss, train_acc, train_auc = self.train_epoch(
                model, train_loader, optimizer, criterion
            )

            # Validate
            val_metrics = self.validate(model, val_loader, criterion)

            # Update scheduler
            scheduler.step()

            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_auc'].append(train_auc)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_auc'].append(val_metrics['auc'])
            history['val_f1'].append(val_metrics['f1'])

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{self.config['num_epochs']} | "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                      f"AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")

            # Save best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_epoch = epoch
                torch.save(model.state_dict(), self.save_dir / f'fold_{fold+1}_best.pt')

        # Load best model and get final validation metrics
        model.load_state_dict(torch.load(self.save_dir / f'fold_{fold+1}_best.pt'))
        final_metrics = self.validate(model, val_loader, criterion)

        print(f"\nFold {fold+1} Best Results (Epoch {best_epoch+1}):")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  AUC: {final_metrics['auc']:.4f}")
        print(f"  F1: {final_metrics['f1']:.4f}")

        # Store results
        fold_result = {
            'fold': fold + 1,
            'best_epoch': best_epoch + 1,
            'history': history,
            'final_metrics': final_metrics
        }

        self.fold_results.append(fold_result)

        return fold_result

    def run_cross_validation(self, fold_data, sleep_labels, adhd_labels, subject_ids):
        """Run k-fold cross-validation"""

        # Stratified K-Fold on subjects
        subject_indices = np.arange(len(fold_data))

        skf = StratifiedKFold(n_splits=self.config['n_folds'], shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(skf.split(subject_indices, adhd_labels)):
            # Create datasets
            train_dataset = FIFDatasetADHD(
                fold_data, sleep_labels, adhd_labels, subject_ids, train_idx.tolist()
            )
            val_dataset = FIFDatasetADHD(
                fold_data, sleep_labels, adhd_labels, subject_ids, val_idx.tolist()
            )

            print(f"\nFold {fold+1}:")
            print(f"  Train: {len(train_idx)} subjects")
            print(f"  Val: {len(val_idx)} subjects")

            # Train fold
            self.train_fold(fold, train_dataset, val_dataset)

        # Aggregate results
        self.aggregate_results()

        # Create plots
        self.create_plots()

        # Save results
        self.save_results()

    def aggregate_results(self):
        """Aggregate results across folds"""
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*80}")

        # Overall metrics
        accuracies = [r['final_metrics']['accuracy'] for r in self.fold_results]
        aucs = [r['final_metrics']['auc'] for r in self.fold_results]
        f1s = [r['final_metrics']['f1'] for r in self.fold_results]

        print(f"\nOverall Metrics (Mean ± Std across {self.config['n_folds']} folds):")
        print(f"  Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

        # Per-class metrics (Control vs ADHD)
        class_names = ['Control', 'ADHD']

        print(f"\nPer-Class Metrics (Mean ± Std):")
        print(f"{'Class':<10} {'Precision':<18} {'Recall':<18} {'F1-Score':<18} {'Support':<10}")
        print("-" * 80)

        for i, class_name in enumerate(class_names):
            precisions = [r['final_metrics']['precision_per_class'][i] for r in self.fold_results]
            recalls = [r['final_metrics']['recall_per_class'][i] for r in self.fold_results]
            f1_scores = [r['final_metrics']['f1_per_class'][i] for r in self.fold_results]
            supports = [r['final_metrics']['support'][i] for r in self.fold_results]

            print(f"{class_name:<10} "
                  f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}   "
                  f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}   "
                  f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}   "
                  f"{int(np.mean(supports)):>6}")

        # Aggregate confusion matrix
        avg_cm = np.mean([r['final_metrics']['confusion_matrix'] for r in self.fold_results], axis=0)
        print(f"\nAverage Confusion Matrix:")
        print("Predicted ->")
        print(f"{'':>15} " + " ".join(f"{c:>10}" for c in class_names))
        for i, class_name in enumerate(class_names):
            print(f"Actual {class_name:<8} " + " ".join(f"{avg_cm[i][j]:>10.1f}" for j in range(2)))

    def create_plots(self):
        """Create plots"""
        class_names = ['Control', 'ADHD']

        # Set style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")

        # 1. Training curves
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Metrics (5-Fold CV)', fontsize=14, fontweight='bold')

        for fold_idx, result in enumerate(self.fold_results):
            ax = axes[fold_idx // 3, fold_idx % 3]
            history = result['history']
            epochs = range(1, len(history['train_loss']) + 1)

            ax.plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
            ax.plot(epochs, history['val_acc'], label='Val Acc', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Accuracy', fontsize=10)
            ax.set_title(f'Fold {fold_idx + 1}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. AUC curves
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Validation AUC Curves (5-Fold CV)', fontsize=14, fontweight='bold')

        for fold_idx, result in enumerate(self.fold_results):
            ax = axes[fold_idx // 3, fold_idx % 3]
            history = result['history']
            epochs = range(1, len(history['val_auc']) + 1)

            ax.plot(epochs, history['val_auc'], linewidth=2, color='#2E86AB')
            ax.axhline(y=result['final_metrics']['auc'], color='r', linestyle='--',
                      label=f"Best: {result['final_metrics']['auc']:.3f}", linewidth=1.5)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('AUC', fontsize=10)
            ax.set_title(f'Fold {fold_idx + 1}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'auc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Confusion matrix
        avg_cm = np.mean([r['final_metrics']['confusion_matrix'] for r in self.fold_results], axis=0)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, ax=ax, annot_kws={'size': 14})
        ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Class', fontsize=12, fontweight='bold')
        ax.set_title('Average Confusion Matrix (5-Fold CV)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Overall metrics box plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        overall_metrics = {
            'Accuracy': [r['final_metrics']['accuracy'] for r in self.fold_results],
            'AUC': [r['final_metrics']['auc'] for r in self.fold_results],
            'F1': [r['final_metrics']['f1'] for r in self.fold_results]
        }

        for idx, (metric_name, values) in enumerate(overall_metrics.items()):
            ax = axes[idx]
            bp = ax.boxplot([values], widths=0.6, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))

            ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric_name} Distribution\n(Mean: {np.mean(values):.4f} ± {np.std(values):.4f})',
                        fontsize=12, fontweight='bold')
            ax.set_xticklabels(['5-Fold CV'])
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1.1])

            # Add individual points
            x = np.random.normal(1, 0.04, size=len(values))
            ax.scatter(x, values, alpha=0.6, s=100, color='darkblue')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'metrics_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nPlots saved to {self.save_dir}/")

    def save_results(self):
        """Save results"""
        summary = {
            'overall_metrics': {
                'accuracy': {
                    'mean': float(np.mean([r['final_metrics']['accuracy'] for r in self.fold_results])),
                    'std': float(np.std([r['final_metrics']['accuracy'] for r in self.fold_results])),
                    'folds': [float(r['final_metrics']['accuracy']) for r in self.fold_results]
                },
                'auc': {
                    'mean': float(np.mean([r['final_metrics']['auc'] for r in self.fold_results])),
                    'std': float(np.std([r['final_metrics']['auc'] for r in self.fold_results])),
                    'folds': [float(r['final_metrics']['auc']) for r in self.fold_results]
                },
                'f1': {
                    'mean': float(np.mean([r['final_metrics']['f1'] for r in self.fold_results])),
                    'std': float(np.std([r['final_metrics']['f1'] for r in self.fold_results])),
                    'folds': [float(r['final_metrics']['f1']) for r in self.fold_results]
                }
            }
        }

        with open(self.save_dir / 'cv_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Results table
        results_df = pd.DataFrame({
            'Fold': [r['fold'] for r in self.fold_results],
            'Best Epoch': [r['best_epoch'] for r in self.fold_results],
            'Accuracy': [r['final_metrics']['accuracy'] for r in self.fold_results],
            'AUC': [r['final_metrics']['auc'] for r in self.fold_results],
            'F1': [r['final_metrics']['f1'] for r in self.fold_results]
        })

        results_df.to_csv(self.save_dir / 'fold_results.csv', index=False)

        print(f"\nResults saved to {self.save_dir}/")


def main():
    """Main function"""

    config = {
        # Data
        'fif_directory': './data/fif_files',  # UPDATE THIS PATH
        'batch_size': 4,
        'num_workers': 0,

        # Cross-validation
        'n_folds': 5,

        # Model
        'step': 4,  # Which step to use (1-5)

        # Training
        'num_epochs': 50,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,

        # Save
        'save_dir': './results/adhd_cv_step4'
    }

    print("=" * 80)
    print("ADHD Classification - 5-Fold Stratified Cross-Validation")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Check if FIF directory exists
    if not Path(config['fif_directory']).exists():
        print(f"\nERROR: FIF directory not found: {config['fif_directory']}")
        print("\nPlease update the 'fif_directory' path in the config")
        print("or create dummy FIF files for testing")
        return

    # Load data
    print("Loading FIF data...")
    try:
        fold_data, sleep_labels, adhd_labels, subject_ids, fold_len = load_fif_data(
            config['fif_directory']
        )
    except Exception as e:
        print(f"\nERROR loading FIF data: {e}")
        print("\nMake sure your FIF directory contains valid .fif files")
        return

    print(f"Loaded {len(fold_data)} subjects")
    print(f"Total epochs: {sum(len(labels) for labels in sleep_labels)}")

    # Run cross-validation
    trainer = CrossValidationTrainerADHD(config)
    trainer.run_cross_validation(fold_data, sleep_labels, adhd_labels, subject_ids)

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
