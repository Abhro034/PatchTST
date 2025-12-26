"""
Data utilities for ADHD classification.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict


class PSGDataset(Dataset):
    """
    Dataset for PSG data.

    Expected data format:
        - X: [N, E, C, L] - PSG signals
        - y: [N] - ADHD labels
        - stage_labels: [N, E] - sleep stage labels (optional)
        - confounds: Dict of [N] arrays (optional)
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stage_labels: Optional[np.ndarray] = None,
        confounds: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Args:
            X: [N, E, C, L] - PSG data
            y: [N] - ADHD labels (0 or 1)
            stage_labels: [N, E] - sleep stage labels (0-4)
            confounds: Dict of confound arrays
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

        self.stage_labels = None
        if stage_labels is not None:
            self.stage_labels = torch.from_numpy(stage_labels).long()

        self.confounds = None
        if confounds is not None:
            self.confounds = {
                k: torch.from_numpy(v).float() if v.dtype == np.float32 or v.dtype == np.float64
                else torch.from_numpy(v).long()
                for k, v in confounds.items()
            }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {
            'X': self.X[idx],
            'y': self.y[idx]
        }

        if self.stage_labels is not None:
            item['stage_labels'] = self.stage_labels[idx]

        if self.confounds is not None:
            item['confounds'] = {k: v[idx] for k, v in self.confounds.items()}

        return item


def create_dummy_data(
    n_subjects: int = 100,
    n_epochs: int = 200,
    n_channels: int = 10,
    seq_len: int = 3000,
    adhd_prevalence: float = 0.3,
    include_stage_labels: bool = False,
    include_confounds: bool = False,
    seed: int = 42
) -> Dict:
    """
    Create dummy PSG data for testing and overfitting checks.

    Args:
        n_subjects: Number of subjects
        n_epochs: Number of epochs per subject
        n_channels: Number of PSG channels
        seq_len: Sequence length per epoch
        adhd_prevalence: Proportion of ADHD subjects
        include_stage_labels: Include sleep stage labels
        include_confounds: Include confound variables
        seed: Random seed

    Returns:
        Dict with 'X', 'y', and optionally 'stage_labels' and 'confounds'
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate PSG data
    X = np.random.randn(n_subjects, n_epochs, n_channels, seq_len).astype(np.float32)

    # Generate labels (stratified)
    y = np.random.rand(n_subjects) < adhd_prevalence
    y = y.astype(np.float32)

    # Add some signal to data (simple pattern for overfitting test)
    for i in range(n_subjects):
        if y[i] == 1:
            # ADHD subjects have higher variance in specific channels
            X[i, :, [0, 2, 5], :] *= 1.5
            # And different frequency content
            X[i, :, [1, 3], :] += 0.5 * np.sin(2 * np.pi * 0.1 * np.arange(seq_len))

    data = {
        'X': X,
        'y': y
    }

    # Generate sleep stage labels
    if include_stage_labels:
        # Simulate sleep stages (0: Wake, 1: N1, 2: N2, 3: N3, 4: REM)
        # Typical sleep progression
        stage_labels = np.zeros((n_subjects, n_epochs), dtype=np.int64)

        for i in range(n_subjects):
            # Early epochs: wake/N1
            stage_labels[i, :20] = np.random.choice([0, 1], size=20, p=[0.7, 0.3])
            # Middle epochs: N2/N3
            stage_labels[i, 20:150] = np.random.choice([2, 3], size=130, p=[0.7, 0.3])
            # Late epochs: N2/REM
            stage_labels[i, 150:] = np.random.choice([2, 4], size=n_epochs-150, p=[0.5, 0.5])

        data['stage_labels'] = stage_labels

    # Generate confounds
    if include_confounds:
        confounds = {}

        # Age (continuous)
        confounds['age'] = np.random.normal(40, 15, n_subjects).astype(np.float32)

        # Sex (binary)
        confounds['sex'] = np.random.choice([0, 1], size=n_subjects)

        # AHI - Apnea-Hypopnea Index (continuous)
        confounds['ahi'] = np.random.exponential(10, n_subjects).astype(np.float32)

        data['confounds'] = confounds

    return data


def create_dataloader(
    dataset: PSGDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader for PSG dataset.

    Args:
        dataset: PSGDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if shuffle else False
    )


def collate_fn(batch):
    """
    Custom collate function for batching PSG data.

    Handles variable-length epochs if needed.
    """
    X = torch.stack([item['X'] for item in batch])
    y = torch.stack([item['y'] for item in batch])

    collated = {
        'X': X,
        'y': y
    }

    # Handle stage labels
    if 'stage_labels' in batch[0]:
        stage_labels = torch.stack([item['stage_labels'] for item in batch])
        collated['stage_labels'] = stage_labels

    # Handle confounds
    if 'confounds' in batch[0]:
        confounds = {}
        for key in batch[0]['confounds'].keys():
            confounds[key] = torch.stack([item['confounds'][key] for item in batch])
        collated['confounds'] = confounds

    return collated


def train_test_split(
    data: Dict,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42
) -> Tuple[Dict, Dict, Dict]:
    """
    Split data into train, validation, and test sets.

    Args:
        data: Dict with 'X', 'y', and optional other fields
        test_size: Proportion for test set
        val_size: Proportion for validation set
        seed: Random seed

    Returns:
        train_data, val_data, test_data
    """
    np.random.seed(seed)

    n_subjects = len(data['y'])
    indices = np.arange(n_subjects)
    np.random.shuffle(indices)

    # Calculate splits
    n_test = int(n_subjects * test_size)
    n_val = int(n_subjects * val_size)
    n_train = n_subjects - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    def split_data(idx):
        split = {
            'X': data['X'][idx],
            'y': data['y'][idx]
        }

        if 'stage_labels' in data:
            split['stage_labels'] = data['stage_labels'][idx]

        if 'confounds' in data:
            split['confounds'] = {
                k: v[idx] for k, v in data['confounds'].items()
            }

        return split

    return split_data(train_idx), split_data(val_idx), split_data(test_idx)
