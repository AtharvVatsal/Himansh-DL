"""
src/utils.py

Shared utilities used by train.py (and later evaluate.py / stats_test.py):
  - reproducibility (seeding)
  - label encoding (datasets don't share a contiguous 0..N-1 label space)
  - accuracy / macro-F1
  - efficiency accounting: param count, FLOPs, inference latency
  - checkpointing
  - append-only results logging to a single CSV, one row per run
    (model x dataset x split_mode x fold/seed)
"""

import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

RESULTS_DIR = Path("results")
RAW_RESULTS_CSV = RESULTS_DIR / "tables" / "raw_results.csv"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic cuDNN -- slightly slower, but seed-to-seed variance should
    # reflect init/data order, not nondeterministic kernel selection
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------
def encode_labels(y_train: np.ndarray, y_test: np.ndarray):
    """
    Raw label spaces differ across datasets: UCI-HAR is 1-indexed (1..6),
    WISDM is 0-indexed via a hand-written map, PAMAP2 activityIDs are
    non-contiguous device codes (e.g. 1,2,3,4,12,13,16,17,...).
    nn.CrossEntropyLoss needs contiguous 0..num_classes-1 targets, so remap
    using classes observed across BOTH train and test (label space is fixed
    dataset metadata, not learned from data -- this is not leakage).
    """
    classes = np.unique(np.concatenate([y_train, y_test]))
    mapping = {c: i for i, c in enumerate(classes)}
    y_train_enc = np.array([mapping[v] for v in y_train], dtype=np.int64)
    y_test_enc = np.array([mapping[v] for v in y_test], dtype=np.int64)
    return y_train_enc, y_test_enc, mapping


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
    )


# ---------------------------------------------------------------------------
# Efficiency accounting
# ---------------------------------------------------------------------------
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model: torch.nn.Module, input_shape: tuple, device: torch.device) -> Optional[float]:
    """
    FLOPs via `thop`. Caveat: thop's RNN/GRU FLOP counting is known to be
    approximate (it does not fully model gate-level matrix multiplies for
    all RNN variants) -- treat this as a consistent, comparable-across-models
    estimate rather than a ground-truth FLOP count. Report this caveat in
    the paper's efficiency section. Returns None if thop fails.
    """
    try:
        import copy
        from thop import profile
        # thop.profile registers `total_ops`/`total_params` buffers directly
        # onto the model it's given, which then breaks state_dict() for
        # checkpointing -- profile a throwaway deep copy instead.
        probe = copy.deepcopy(model).to(device)
        dummy = torch.randn(1, *input_shape).to(device)
        macs, _ = profile(probe, inputs=(dummy,), verbose=False)
        return macs * 2  # MACs -> FLOPs (1 MAC ~= 2 FLOPs)
    except Exception as e:
        print(f"[compute_flops] thop failed ({e}); returning None")
        return None


def measure_latency(model: torch.nn.Module, input_shape: tuple, device: torch.device,
                     batch_sizes=(1, 32), n_warmup: int = 10, n_repeats: int = 50) -> dict:
    """
    Mean inference latency (ms/batch) per batch size, on the given device.
    Run once per architecture (weights don't affect latency), not per seed.
    """
    model = model.to(device).eval()
    results = {}
    with torch.no_grad():
        for bs in batch_sizes:
            dummy = torch.randn(bs, *input_shape).to(device)

            for _ in range(n_warmup):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(n_repeats):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            results[f"latency_ms_bs{bs}"] = (elapsed / n_repeats) * 1000
    return results


def profile_model(model: torch.nn.Module, input_shape: tuple, device: torch.device) -> dict:
    """Convenience wrapper bundling params + FLOPs + latency into one dict."""
    stats = dict(params=count_parameters(model))
    stats["flops"] = compute_flops(model, input_shape, device)
    stats.update(measure_latency(model, input_shape, device))
    return stats


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Stops training when monitored value (lower=better, e.g. val loss)
    hasn't improved for `patience` epochs. Tracks best state_dict in memory."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float("inf")
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, value: float, model: torch.nn.Module):
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: torch.nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(model: torch.nn.Module, path: Path, extra: Optional[dict] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: Path, map_location=None):
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    return payload


# ---------------------------------------------------------------------------
# Results logging (single append-only CSV, one row per run)
# ---------------------------------------------------------------------------
def append_result_row(row: dict, csv_path: Path = RAW_RESULTS_CSV):
    """
    Appends one row to results/tables/raw_results.csv, creating it with a
    header on first write. Expected keys (extend as needed, missing keys
    become NaN via pandas):
        model, dataset, split_mode, fold_id, seed, accuracy, macro_f1,
        params, flops, latency_ms_bs1, latency_ms_bs32
    This raw log is the input to the later master_comparison.csv aggregation
    and to stats_test.py's per-fold significance testing.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)