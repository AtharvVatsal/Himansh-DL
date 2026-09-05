"""
src/data_loader.py

Loads the processed (X, y, subject) arrays produced by preprocessing.py and
exposes two split modes, both required per the evaluation protocol:

  1. Official train/test split   -> get_official_split()
  2. Subject-independent k-fold  -> get_loso_folds()   [headline result]

Normalization is deliberately NOT done in preprocessing.py -- it happens here,
per split/fold, fit on the TRAIN portion only, to avoid leakage. Every
training script should go through build_dataloaders() rather than reading
the .npy files directly.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional

from src.preprocessing import normalize_channels, PROCESSED_DIR

# Number of subject-groups to fold into for LOSO CV. True leave-ONE-out is used
# where the subject count is small (PAMAP2); larger subject pools are grouped
# into folds for tractable runtime. Document/justify in the report if changed.
LOSO_N_FOLDS = {
    "ucihar": 6,   # 30 subjects -> ~5 subjects held out per fold
    "wisdm": 6,    # 36 subjects -> ~6 subjects held out per fold
    "pamap2": 9,   # 9 subjects  -> true leave-one-subject-out
}

# Fixed random seed for any split that isn't dictated by an official protocol
# (subject-level shuffling only -- never shuffles within a subject's windows).
SPLIT_SEED = 42


# ---------------------------------------------------------------------------
# Raw array loading
# ---------------------------------------------------------------------------
def _load_processed(dataset: str, harmonized: bool = False):
    suffix = "_harmonized" if harmonized else ""
    d = PROCESSED_DIR / dataset
    X = np.load(d / f"X.npy")
    y = np.load(d / f"y.npy")
    subject = np.load(d / f"subject.npy")
    official_split = None
    p = d / f"official_split.npy"
    if p.exists():
        official_split = np.load(p, allow_pickle=True)
    return X, y, subject, official_split


class HARDataset(Dataset):
    """Simple (X, y) wrapper. X: (N, T, C) float32, y: (N,) int64."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Split 1: official train/test
# ---------------------------------------------------------------------------
def get_official_split(dataset: str, seed: int = SPLIT_SEED):
    """
    Returns (X_train, y_train, X_test, y_test), normalized using train-fold
    statistics.

    - UCI-HAR: uses the dataset's real official split (saved by preprocessing.py).
    - WISDM / PAMAP2: no official split is published, so we do a fixed,
      seeded, SUBJECT-LEVEL 80/20 split (never split within a subject).
      This is a documented substitute, not a claimed "official" protocol --
      say so explicitly in the report's evaluation section.
    """
    X, y, subject, official_split = _load_processed(dataset)

    if official_split is not None:
        train_mask = official_split == "train"
        test_mask = official_split == "test"
    else:
        rng = np.random.RandomState(seed)
        unique_subjects = np.unique(subject)
        rng.shuffle(unique_subjects)
        n_test = max(1, int(round(0.2 * len(unique_subjects))))
        test_subjects = set(unique_subjects[:n_test])
        train_mask = np.array([s not in test_subjects for s in subject])
        test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    X_train, X_test, mean, std = normalize_channels(X_train, X_test)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Split 2: subject-independent (LOSO) k-fold -- the headline evaluation
# ---------------------------------------------------------------------------
def get_loso_folds(dataset: str, n_folds: Optional[int] = None, seed: int = SPLIT_SEED):
    """
    Generator yielding one dict per fold:
        {fold_id, held_out_subjects, X_train, y_train, X_test, y_test}

    Subjects are randomly (seed-fixed) partitioned into n_folds disjoint
    groups; each fold holds one group out entirely (leave-N-subjects-out).
    Normalization is fit on that fold's train subjects only.
    """
    X, y, subject, _ = _load_processed(dataset)
    n_folds = n_folds or LOSO_N_FOLDS[dataset]

    rng = np.random.RandomState(seed)
    unique_subjects = np.unique(subject)
    rng.shuffle(unique_subjects)
    subject_groups = np.array_split(unique_subjects, n_folds)

    for fold_id, held_out in enumerate(subject_groups):
        held_out_set = set(held_out.tolist())
        test_mask = np.array([s in held_out_set for s in subject])
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        X_train, X_test, mean, std = normalize_channels(X_train, X_test)

        yield dict(
            fold_id=fold_id,
            held_out_subjects=held_out.tolist(),
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )


# ---------------------------------------------------------------------------
# Convenience: DataLoader builder
# ---------------------------------------------------------------------------
def build_dataloaders(X_train, y_train, X_test, y_test,
                       batch_size: int = 64, num_workers: int = 0):
    train_loader = DataLoader(
        HARDataset(X_train, y_train), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=False,
    )
    test_loader = DataLoader(
        HARDataset(X_test, y_test), batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    # Smoke test -- run after preprocessing.py has populated data/processed/
    for ds in ("ucihar", "wisdm", "pamap2"):
        try:
            X_tr, y_tr, X_te, y_te = get_official_split(ds)
            print(f"[{ds}] official split -> train {X_tr.shape}, test {X_te.shape}")

            n_folds = 0
            for fold in get_loso_folds(ds):
                n_folds += 1
            print(f"[{ds}] LOSO -> {n_folds} folds")
        except FileNotFoundError as e:
            print(f"[{ds}] skipped, processed data not found yet: {e}")