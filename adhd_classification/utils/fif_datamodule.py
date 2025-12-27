"""
Data module for loading FIF (FIFF) files for ADHD Classification
Compatible with MNE-Python epoch files
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
from pathlib import Path


def process_subject_fif(fif_path):
    """
    Process a single subject from .fif file

    Returns:
        patient_eeg: (n_epochs, n_channels, n_timepoints)
        sleep_stages: (n_epochs,) - sleep stage labels
        adhd_label: int - ADHD label for the subject
    """
    import mne

    stage_map = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'R': 4}

    # Load data
    raw = mne.read_epochs(fif_path, preload=True, verbose=False)
    patient_eeg = raw.get_data()  # (n_epochs, n_channels, n_timepoints)

    patient_eeg = patient_eeg[:, :, :3000]  # Truncate to 30 sec at 100 Hz

    # Get ADHD label
    if hasattr(raw, 'metadata') and raw.metadata is not None and 'ADHD' in raw.metadata.columns:
        adhd_label = int(raw.metadata['ADHD'].iloc[0])
    else:
        # Fallback: check filename or use default
        adhd_label = 0

    # Get sleep stages (optional for ADHD classification)
    try:
        sleep_stages = raw.events[:, 2]
        if sleep_stages.dtype == object:
            sleep_stages = np.array([stage_map.get(s, 0) for s in sleep_stages])
        else:
            sleep_stages = np.array([stage_map.get(int(s), int(s)) for s in sleep_stages])
    except:
        sleep_stages = np.zeros(len(patient_eeg), dtype=np.int64)

    # Normalize each epoch with robust handling
    normalized_eeg = []
    for epoch in patient_eeg:
        # Check for invalid values first
        if np.isnan(epoch).any() or np.isinf(epoch).any():
            # Replace NaN/Inf with median
            epoch = np.nan_to_num(epoch, nan=np.nanmedian(epoch), posinf=np.nanmedian(epoch), neginf=np.nanmedian(epoch))

        # Per-channel normalization with robust epsilon
        mean = epoch.mean(axis=1, keepdims=True)
        std = epoch.std(axis=1, keepdims=True)

        # Use larger epsilon and clip extreme values
        epsilon = 1e-6
        normalized_epoch = (epoch - mean) / (std + epsilon)

        # Clip to prevent extreme values
        normalized_epoch = np.clip(normalized_epoch, -10, 10)

        normalized_eeg.append(normalized_epoch)

    patient_eeg = np.array(normalized_eeg, dtype=np.float32)

    return patient_eeg, sleep_stages, adhd_label


def load_fif_data(fif_directory):
    """
    Load all .fif files from directory

    Returns:
        fold_data: List of (n_epochs, n_channels, n_samples) arrays per subject
        sleep_labels: List of sleep stage arrays per subject
        adhd_labels: Array of ADHD labels per subject
        subject_ids: List of subject IDs
        fold_len: Array of number of epochs per subject
    """
    fif_dir = Path(fif_directory)
    fif_files = sorted(list(fif_dir.glob('*.fif')))

    if len(fif_files) == 0:
        raise ValueError(f"No .fif files found in {fif_directory}")

    print(f"Found {len(fif_files)} .fif files")

    fold_data = []
    sleep_labels = []
    adhd_labels = []
    subject_ids = []

    for fif_path in tqdm(fif_files, desc="Loading .fif files"):
        try:
            epoch_data, sleep_stages, adhd_label = process_subject_fif(fif_path)

            fold_data.append(epoch_data)
            sleep_labels.append(sleep_stages)
            adhd_labels.append(adhd_label)
            subject_ids.append(fif_path.stem)

        except Exception as e:
            print(f"Error processing {fif_path.name}: {e}")
            continue

    adhd_labels = np.array(adhd_labels)
    fold_len = np.array([len(s) for s in sleep_labels])

    print(f"Loaded {len(fold_data)} subjects successfully")
    print(f"Total epochs: {sum(fold_len)}")
    print(f"ADHD: {np.sum(adhd_labels)}, Control: {len(adhd_labels) - np.sum(adhd_labels)}")

    return fold_data, sleep_labels, adhd_labels, subject_ids, fold_len


class FIFDatasetADHD(Dataset):
    """
    PyTorch Dataset for FIF files with ADHD classification

    Returns subject-level data (all epochs for one subject)
    """
    def __init__(
        self,
        fold_data: List[np.ndarray],
        sleep_labels: List[np.ndarray],
        adhd_labels: np.ndarray,
        subject_ids: List[str],
        indices: Optional[List[int]] = None
    ):
        """
        Args:
            fold_data: List of (n_epochs, n_channels, n_samples) arrays
            sleep_labels: List of sleep stage arrays
            adhd_labels: Array of ADHD labels
            subject_ids: List of subject IDs
            indices: Optional list of indices to use (for train/val/test split)
        """
        self.fold_data = fold_data
        self.sleep_labels = sleep_labels
        self.adhd_labels = adhd_labels
        self.subject_ids = subject_ids

        # Use subset of indices if provided
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(fold_data)))

        # Get dimensions from first subject
        first_data = fold_data[self.indices[0]]
        self.n_channels = first_data.shape[1]
        self.n_samples = first_data.shape[2]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get all epochs for a single subject

        Returns:
            Dict with:
                - X: [E, N, L] - all epochs for this subject
                - y: scalar - ADHD label
                - subject_id: str
                - sleep_stages: [E] - sleep stages (optional)
        """
        subj_idx = self.indices[idx]

        # Get all epochs for this subject: (n_epochs, n_channels, n_samples)
        X = torch.FloatTensor(self.fold_data[subj_idx])  # [E, N, L]

        # Get ADHD label (subject-level)
        y = torch.FloatTensor([self.adhd_labels[subj_idx]])  # [1]

        # Get sleep stages (optional)
        sleep_stages = torch.LongTensor(self.sleep_labels[subj_idx])  # [E]

        subject_id = self.subject_ids[subj_idx]

        return {
            'X': X,
            'y': y,
            'subject_id': subject_id,
            'sleep_stages': sleep_stages
        }


def collate_fn_adhd(batch, num_epochs_sample=128, sampling_strategy='uniform'):
    """
    Custom collate function with epoch sampling for memory efficiency

    Args:
        batch: List of subject data dicts
        num_epochs_sample: Number of epochs to sample per subject (default: 128)
        sampling_strategy: 'uniform', 'random', or 'stage_stratified'

    Returns:
        Batched data with sampled epochs per subject
    """
    batched_X = []
    batched_y = []
    batched_ids = []
    batched_stages = []

    for item in batch:
        X = item['X']  # [E, N, L]
        n_epochs = X.shape[0]

        if n_epochs <= num_epochs_sample:
            # Use all epochs if fewer than target
            sampled_X = X
            sampled_stages = item['sleep_stages']
        else:
            # Sample epochs based on strategy
            if sampling_strategy == 'uniform':
                # Uniformly sample across the night
                step = n_epochs / num_epochs_sample
                indices = torch.tensor([int(i * step) for i in range(num_epochs_sample)])

            elif sampling_strategy == 'random':
                # Random sampling (default behavior)
                indices = torch.randperm(n_epochs)[:num_epochs_sample]

            elif sampling_strategy == 'stage_stratified':
                # Sample proportionally from each sleep stage
                stages = item['sleep_stages']
                indices = []

                # Get unique stages and their counts
                unique_stages = torch.unique(stages)
                epochs_per_stage = num_epochs_sample // len(unique_stages)

                for stage in unique_stages:
                    stage_indices = torch.where(stages == stage)[0]
                    if len(stage_indices) > 0:
                        n_sample = min(epochs_per_stage, len(stage_indices))
                        sampled = stage_indices[torch.randperm(len(stage_indices))[:n_sample]]
                        indices.extend(sampled.tolist())

                # If we haven't reached target, randomly sample remaining
                while len(indices) < num_epochs_sample:
                    remaining = num_epochs_sample - len(indices)
                    all_indices = set(range(n_epochs))
                    available = list(all_indices - set(indices))
                    if not available:
                        break
                    additional = np.random.choice(available, min(remaining, len(available)), replace=False)
                    indices.extend(additional.tolist())

                indices = torch.tensor(indices[:num_epochs_sample])

            else:
                raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

            sampled_X = X[indices]
            sampled_stages = item['sleep_stages'][indices]

        batched_X.append(sampled_X)
        batched_y.append(item['y'])
        batched_ids.append(item['subject_id'])
        batched_stages.append(sampled_stages)

    return {
        'X': torch.stack(batched_X),  # [B, E_sampled, N, L]
        'y': torch.stack(batched_y).squeeze(-1),  # [B]
        'subject_ids': batched_ids,
        'sleep_stages': torch.stack(batched_stages),  # [B, E_sampled]
        'num_epochs_sampled': num_epochs_sample
    }


def train_test_split_subjects(
    n_subjects: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split subject indices into train/val/test

    Returns:
        train_indices, val_indices, test_indices
    """
    np.random.seed(seed)

    indices = np.arange(n_subjects)
    np.random.shuffle(indices)

    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)

    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train+n_val].tolist()
    test_idx = indices[n_train+n_val:].tolist()

    return train_idx, val_idx, test_idx


def evaluate_subject_full(model, subject_data, device, num_epochs_per_batch=128):
    """
    Evaluate a subject by aggregating predictions over all epochs

    Args:
        model: ADHDClassifier model
        subject_data: Dict with 'X' [E, N, L], 'y', 'subject_id'
        device: torch device
        num_epochs_per_batch: Number of epochs to process at once

    Returns:
        prediction: Subject-level prediction
        probability: Subject-level probability
    """
    model.eval()

    X_all = subject_data['X']  # [E, N, L]
    n_epochs = X_all.shape[0]

    # Process in chunks
    epoch_embeddings = []

    with torch.no_grad():
        for start_idx in range(0, n_epochs, num_epochs_per_batch):
            end_idx = min(start_idx + num_epochs_per_batch, n_epochs)
            X_chunk = X_all[start_idx:end_idx]  # [chunk_size, N, L]

            # Add batch dimension
            X_batch = X_chunk.unsqueeze(0).to(device)  # [1, chunk_size, N, L]

            # Get epoch embeddings (before night pooling)
            output = model(X_batch, return_intermediates=True)
            u = output['intermediates']['u']  # [1, chunk_size, D]

            epoch_embeddings.append(u.squeeze(0).cpu())  # [chunk_size, D]

    # Concatenate all epoch embeddings
    all_u = torch.cat(epoch_embeddings, dim=0)  # [E, D]

    # Apply night pooling manually
    # Use mean pooling for aggregation
    z = all_u.mean(dim=0, keepdim=True).to(device)  # [1, D]

    # Final classification
    logit = model.classifier(z).cpu().item()
    prob = torch.sigmoid(torch.tensor(logit)).item()
    pred = 1 if prob > 0.5 else 0

    return pred, prob


def evaluate_subject_multisampling(model, subject_data, device,
                                   num_samples=5, num_epochs_per_sample=128):
    """
    Evaluate subject using multiple random epoch samplings and average

    Args:
        model: ADHDClassifier model
        subject_data: Dict with 'X' [E, N, L], 'y', 'subject_id'
        device: torch device
        num_samples: Number of random samplings
        num_epochs_per_sample: Epochs per sampling

    Returns:
        prediction: Averaged subject-level prediction
        probability: Averaged subject-level probability
    """
    model.eval()

    X_all = subject_data['X']  # [E, N, L]
    n_epochs = X_all.shape[0]

    probabilities = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Random sample
            if n_epochs > num_epochs_per_sample:
                indices = torch.randperm(n_epochs)[:num_epochs_per_sample]
                X_sample = X_all[indices]
            else:
                X_sample = X_all

            # Add batch dimension
            X_batch = X_sample.unsqueeze(0).to(device)  # [1, E_sampled, N, L]

            # Forward pass
            output = model(X_batch)
            logit = output['logits'].squeeze().cpu().item()
            prob = torch.sigmoid(torch.tensor(logit)).item()

            probabilities.append(prob)

    # Average probabilities
    avg_prob = np.mean(probabilities)
    pred = 1 if avg_prob > 0.5 else 0

    return pred, avg_prob


def create_collate_fn(num_epochs_sample=128, sampling_strategy='uniform'):
    """
    Create a collate function with fixed parameters

    Args:
        num_epochs_sample: Number of epochs to sample per subject
        sampling_strategy: 'uniform', 'random', or 'stage_stratified'

    Returns:
        Collate function for DataLoader
    """
    def collate_wrapper(batch):
        return collate_fn_adhd(batch, num_epochs_sample, sampling_strategy)
    return collate_wrapper


def create_dataloaders(
    fif_directory: str,
    batch_size: int = 8,
    num_workers: int = 4,
    num_epochs_sample: int = 128,
    sampling_strategy: str = 'uniform',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train/val/test dataloaders for ADHD classification

    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    # Load data
    print("Loading FIF data...")
    fold_data, sleep_labels, adhd_labels, subject_ids, fold_len = load_fif_data(fif_directory)

    # Split subjects
    n_subjects = len(fold_data)
    train_idx, val_idx, test_idx = train_test_split_subjects(
        n_subjects, train_ratio, val_ratio, test_ratio, seed
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_idx)} subjects")
    print(f"  Val: {len(val_idx)} subjects")
    print(f"  Test: {len(test_idx)} subjects")

    # Create datasets
    train_dataset = FIFDatasetADHD(fold_data, sleep_labels, adhd_labels, subject_ids, train_idx)
    val_dataset = FIFDatasetADHD(fold_data, sleep_labels, adhd_labels, subject_ids, val_idx)
    test_dataset = FIFDatasetADHD(fold_data, sleep_labels, adhd_labels, subject_ids, test_idx)

    # Create collate function with epoch sampling
    collate_fn = create_collate_fn(num_epochs_sample, sampling_strategy)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Metadata
    metadata = {
        'n_subjects': n_subjects,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'n_channels': train_dataset.n_channels,
        'seq_len': train_dataset.n_samples,
        'adhd_count': int(np.sum(adhd_labels)),
        'control_count': int(len(adhd_labels) - np.sum(adhd_labels)),
        'num_epochs_sample': num_epochs_sample,
        'sampling_strategy': sampling_strategy
    }

    return train_loader, val_loader, test_loader, metadata
