# """
# src/train.py

# Generic training driver for all 4 models (only bigru_baseline is registered
# so far -- add bilstm_attention_baseline, cnn_bigru_reference, and
# dual_attention_bigru to MODEL_REGISTRY as they're implemented in later weeks;
# nothing else in this file needs to change).

# Supports both evaluation protocols from the plan:
#   --split_mode official  -> single train/test split, run N seeds
#   --split_mode loso       -> subject-independent k-fold, run N seeds per fold

# Every run (one seed on one split/fold) appends a row to
# results/tables/raw_results.csv via utils.append_result_row. Efficiency
# metrics (params/FLOPs/latency) are computed once per architecture, not
# per seed, and attached to every row for that architecture.

# Example:
#     python -m src.train --model bigru_baseline --dataset ucihar --split_mode official --seeds 0 1 2
#     python -m src.train --model bigru_baseline --dataset pamap2 --split_mode loso --seeds 0 1 2 --epochs 40
# """

# import argparse
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader

# from src.data_loader import get_official_split, get_loso_folds, build_dataloaders, HARDataset
# from src.models.bigru_baseline import BiGRUBaseline
# from src.utils import (
#     set_seed, get_device, encode_labels, compute_metrics,
#     profile_model, EarlyStopping, save_checkpoint, append_result_row,
# )

# # ---------------------------------------------------------------------------
# # Model registry -- add new models here as they're implemented (Weeks 5-7)
# # ---------------------------------------------------------------------------
# MODEL_REGISTRY = {
#     "bigru_baseline": BiGRUBaseline,
#     # "bilstm_attention_baseline": BiLSTMAttentionBaseline,   # Week 5
#     # "cnn_bigru_reference": CNNBiGRUReference,               # Week 6
#     # "dual_attention_bigru": DualAttentionBiGRU,             # Week 7
# }


# def build_model(model_name: str, input_size: int, num_classes: int) -> nn.Module:
#     if model_name not in MODEL_REGISTRY:
#         raise ValueError(f"Unknown model '{model_name}'. Registered: {list(MODEL_REGISTRY)}")
#     return MODEL_REGISTRY[model_name](input_size=input_size, num_classes=num_classes)


# # ---------------------------------------------------------------------------
# # Single train/eval run (one seed, one split or fold)
# # ---------------------------------------------------------------------------
# def train_one_run(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
#                    device: torch.device, epochs: int, lr: float, patience: int,
#                    val_fraction: float = 0.1) -> dict:
#     """
#     Trains `model` with a small held-out slice of the TRAIN set used only for
#     early stopping (not subject-independent -- a pragmatic simplification for
#     early-stopping control; final metrics are always computed on the real
#     held-out test_loader, which IS subject-independent under LOSO).
#     """
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()

#     # carve a validation slice out of the training dataset for early stopping
#     full_train_ds = train_loader.dataset
#     n_val = max(1, int(len(full_train_ds) * val_fraction))
#     n_train = len(full_train_ds) - n_val
#     train_ds, val_ds = torch.utils.data.random_split(
#         full_train_ds, [n_train, n_val],
#         generator=torch.Generator().manual_seed(0),
#     )
#     train_sub_loader = DataLoader(train_ds, batch_size=train_loader.batch_size, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=train_loader.batch_size, shuffle=False)

#     stopper = EarlyStopping(patience=patience)

#     for epoch in range(epochs):
#         model.train()
#         for xb, yb in train_sub_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             optimizer.zero_grad()
#             loss = criterion(model(xb), yb)
#             loss.backward()
#             optimizer.step()

#         model.eval()
#         val_losses = []
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 xb, yb = xb.to(device), yb.to(device)
#                 val_losses.append(criterion(model(xb), yb).item())
#         val_loss = float(np.mean(val_losses))

#         stopper.step(val_loss, model)
#         if stopper.should_stop:
#             print(f"    early stop at epoch {epoch+1} (best val_loss={stopper.best_value:.4f})")
#             break

#     stopper.restore_best(model)

#     # final evaluation on the real (subject-independent under LOSO) test set
#     model.eval()
#     all_preds, all_true = [], []
#     with torch.no_grad():
#         for xb, yb in test_loader:
#             xb = xb.to(device)
#             preds = model(xb).argmax(dim=1).cpu().numpy()
#             all_preds.append(preds)
#             all_true.append(yb.numpy())
#     y_pred = np.concatenate(all_preds)
#     y_true = np.concatenate(all_true)
#     return compute_metrics(y_true, y_pred)


# # ---------------------------------------------------------------------------
# # Driver
# # ---------------------------------------------------------------------------
# def run(args):
#     device = get_device()
#     print(f"device: {device}")

#     # Efficiency profiling done once per architecture, before any training,
#     # using the input shape from one representative split.
#     X_probe, y_probe, _, _ = get_official_split(args.dataset)
#     input_shape = X_probe.shape[1:]  # (T, C)

#     probe_model = build_model(args.model, input_size=input_shape[-1], num_classes=len(np.unique(y_probe)))
#     efficiency = profile_model(probe_model, input_shape, device)
#     print(f"efficiency profile ({args.model} on {args.dataset}): {efficiency}")

#     if args.split_mode == "official":
#         _run_official(args, device, input_shape, efficiency)
#     elif args.split_mode == "loso":
#         _run_loso(args, device, efficiency)
#     else:
#         raise ValueError(args.split_mode)


# def _run_official(args, device, input_shape, efficiency):
#     X_train, y_train, X_test, y_test = get_official_split(args.dataset)
#     y_train_enc, y_test_enc, mapping = encode_labels(y_train, y_test)
#     num_classes = len(mapping)

#     for seed in args.seeds:
#         print(f"[{args.model}/{args.dataset}/official] seed={seed}")
#         set_seed(seed)
#         model = build_model(args.model, input_size=input_shape[-1], num_classes=num_classes)
#         train_loader, test_loader = build_dataloaders(
#             X_train, y_train_enc, X_test, y_test_enc, batch_size=args.batch_size,
#         )
#         metrics = train_one_run(model, train_loader, test_loader, device,
#                                  epochs=args.epochs, lr=args.lr, patience=args.patience)
#         print(f"    -> {metrics}")

#         ckpt_path = Path("checkpoints") / args.model / f"{args.dataset}_official_seed{seed}.pt"
#         save_checkpoint(model, ckpt_path)

#         append_result_row({
#             "model": args.model, "dataset": args.dataset, "split_mode": "official",
#             "fold_id": None, "seed": seed, **metrics, **efficiency,
#         })


# def _run_loso(args, device, efficiency):
#     for fold in get_loso_folds(args.dataset, n_folds=args.n_folds):
#         fold_id = fold["fold_id"]
#         y_train_enc, y_test_enc, mapping = encode_labels(fold["y_train"], fold["y_test"])
#         num_classes = len(mapping)
#         input_shape = fold["X_train"].shape[1:]

#         for seed in args.seeds:
#             print(f"[{args.model}/{args.dataset}/loso] fold={fold_id} "
#                   f"held_out={fold['held_out_subjects']} seed={seed}")
#             set_seed(seed)
#             model = build_model(args.model, input_size=input_shape[-1], num_classes=num_classes)
#             train_loader, test_loader = build_dataloaders(
#                 fold["X_train"], y_train_enc, fold["X_test"], y_test_enc,
#                 batch_size=args.batch_size,
#             )
#             metrics = train_one_run(model, train_loader, test_loader, device,
#                                      epochs=args.epochs, lr=args.lr, patience=args.patience)
#             print(f"    -> {metrics}")

#             ckpt_path = Path("checkpoints") / args.model / f"{args.dataset}_loso_fold{fold_id}_seed{seed}.pt"
#             save_checkpoint(model, ckpt_path)

#             append_result_row({
#                 "model": args.model, "dataset": args.dataset, "split_mode": "loso",
#                 "fold_id": fold_id, "seed": seed, **metrics, **efficiency,
#             })


# # ---------------------------------------------------------------------------
# # CLI
# # ---------------------------------------------------------------------------
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
#     p.add_argument("--dataset", required=True, choices=["ucihar", "wisdm", "pamap2"])
#     p.add_argument("--split_mode", required=True, choices=["official", "loso"])
#     p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
#     p.add_argument("--n_folds", type=int, default=None, help="override LOSO_N_FOLDS default")
#     p.add_argument("--epochs", type=int, default=50)
#     p.add_argument("--batch_size", type=int, default=64)
#     p.add_argument("--lr", type=float, default=1e-3)
#     p.add_argument("--patience", type=int, default=8)
#     return p.parse_args()


# if __name__ == "__main__":
#     run(parse_args())


"""
src/train.py

Generic training driver for all 4 models (only bigru_baseline is registered
so far -- add bilstm_attention_baseline, cnn_bigru_reference, and
dual_attention_bigru to MODEL_REGISTRY as they're implemented in later weeks;
nothing else in this file needs to change).

Supports both evaluation protocols from the plan:
  --split_mode official  -> single train/test split, run N seeds
  --split_mode loso       -> subject-independent k-fold, run N seeds per fold

Every run (one seed on one split/fold) appends a row to
results/tables/raw_results.csv via utils.append_result_row. Efficiency
metrics (params/FLOPs/latency) are computed once per architecture, not
per seed, and attached to every row for that architecture.

Example:
    python -m src.train --model bigru_baseline --dataset ucihar --split_mode official --seeds 0 1 2
    python -m src.train --model bigru_baseline --dataset pamap2 --split_mode loso --seeds 0 1 2 --epochs 40
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_loader import get_official_split, get_loso_folds, build_dataloaders, HARDataset
from src.models.bigru_baseline import BiGRUBaseline
from src.models.bilstm_attention_baseline import BiLSTMAttentionBaseline
from src.models.cnn_bigru_reference import CNNBiGRUReference
from src.models.dual_attention_bigru import DualAttentionBiGRU
from src.preprocessing import NATIVE_CONFIGS
from src.utils import (
    set_seed, get_device, encode_labels, compute_metrics,
    profile_model, EarlyStopping, save_checkpoint, append_result_row,
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "bigru_baseline": BiGRUBaseline,
    "bilstm_attention_baseline": BiLSTMAttentionBaseline,
    "cnn_bigru_reference": CNNBiGRUReference,
    "dual_attention_bigru": DualAttentionBiGRU,
}

# Ablation arms for the proposed model, selected via --dual_attn_variant.
# "full" is the proposed model (spectral-gated dual attention); the others
# isolate each component's contribution -- "plain" reproduces the already-
# closed literature gap (learned channel + temporal attention, no spectral
# prior), so the delta between "plain" and "full" is this project's headline
# architectural-novelty result.
DUAL_ATTN_VARIANTS = {
    "full":            dict(use_channel_attention=True,  use_temporal_attention=True,  use_spectral_gate=True),
    "plain":           dict(use_channel_attention=True,  use_temporal_attention=True,  use_spectral_gate=False),
    "channel_only":    dict(use_channel_attention=True,  use_temporal_attention=False, use_spectral_gate=True),
    "temporal_only":   dict(use_channel_attention=False, use_temporal_attention=True,  use_spectral_gate=True),
}


def build_model(model_name: str, input_size: int, num_classes: int,
                 dataset: str = None, dual_attn_variant: str = "full") -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Registered: {list(MODEL_REGISTRY)}")
    if model_name == "dual_attention_bigru":
        sampling_rate_hz = NATIVE_CONFIGS[dataset]["sampling_rate"]
        variant_flags = DUAL_ATTN_VARIANTS[dual_attn_variant]
        return DualAttentionBiGRU(
            input_size=input_size, num_classes=num_classes,
            sampling_rate_hz=sampling_rate_hz, **variant_flags,
        )
    return MODEL_REGISTRY[model_name](input_size=input_size, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Single train/eval run (one seed, one split or fold)
# ---------------------------------------------------------------------------
def train_one_run(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                   device: torch.device, epochs: int, lr: float, patience: int,
                   val_fraction: float = 0.1) -> dict:
    """
    Trains `model` with a small held-out slice of the TRAIN set used only for
    early stopping (not subject-independent -- a pragmatic simplification for
    early-stopping control; final metrics are always computed on the real
    held-out test_loader, which IS subject-independent under LOSO).
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # carve a validation slice out of the training dataset for early stopping
    full_train_ds = train_loader.dataset
    n_val = max(1, int(len(full_train_ds) * val_fraction))
    n_train = len(full_train_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    train_sub_loader = DataLoader(train_ds, batch_size=train_loader.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_loader.batch_size, shuffle=False)

    stopper = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_sub_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = float(np.mean(val_losses))

        stopper.step(val_loss, model)
        if stopper.should_stop:
            print(f"    early stop at epoch {epoch+1} (best val_loss={stopper.best_value:.4f})")
            break

    stopper.restore_best(model)

    # final evaluation on the real (subject-independent under LOSO) test set
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    return compute_metrics(y_true, y_pred)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(args):
    device = get_device()
    print(f"device: {device}")

    # Efficiency profiling done once per architecture, before any training,
    # using the input shape from one representative split.
    X_probe, y_probe, _, _ = get_official_split(args.dataset)
    input_shape = X_probe.shape[1:]  # (T, C)

    probe_model = build_model(args.model, input_size=input_shape[-1], num_classes=len(np.unique(y_probe)),
                               dataset=args.dataset, dual_attn_variant=args.dual_attn_variant)
    efficiency = profile_model(probe_model, input_shape, device)
    print(f"efficiency profile ({args.model} on {args.dataset}): {efficiency}")

    if args.split_mode == "official":
        _run_official(args, device, input_shape, efficiency)
    elif args.split_mode == "loso":
        _run_loso(args, device, efficiency)
    else:
        raise ValueError(args.split_mode)


def _run_official(args, device, input_shape, efficiency):
    X_train, y_train, X_test, y_test = get_official_split(args.dataset)
    y_train_enc, y_test_enc, mapping = encode_labels(y_train, y_test)
    num_classes = len(mapping)

    for seed in args.seeds:
        print(f"[{args.model}/{args.dataset}/official] seed={seed}")
        set_seed(seed)
        model = build_model(args.model, input_size=input_shape[-1], num_classes=num_classes,
                             dataset=args.dataset, dual_attn_variant=args.dual_attn_variant)
        train_loader, test_loader = build_dataloaders(
            X_train, y_train_enc, X_test, y_test_enc, batch_size=args.batch_size,
        )
        metrics = train_one_run(model, train_loader, test_loader, device,
                                 epochs=args.epochs, lr=args.lr, patience=args.patience)
        print(f"    -> {metrics}")

        ckpt_path = Path("checkpoints") / args.model / f"{args.dataset}_official_seed{seed}.pt"
        save_checkpoint(model, ckpt_path)

        append_result_row({
            "model": args.model, "dataset": args.dataset, "split_mode": "official",
            "fold_id": None, "seed": seed, **metrics, **efficiency,
        })


def _run_loso(args, device, efficiency):
    for fold in get_loso_folds(args.dataset, n_folds=args.n_folds):
        fold_id = fold["fold_id"]
        y_train_enc, y_test_enc, mapping = encode_labels(fold["y_train"], fold["y_test"])
        num_classes = len(mapping)
        input_shape = fold["X_train"].shape[1:]

        for seed in args.seeds:
            print(f"[{args.model}/{args.dataset}/loso] fold={fold_id} "
                  f"held_out={fold['held_out_subjects']} seed={seed}")
            set_seed(seed)
            model = build_model(args.model, input_size=input_shape[-1], num_classes=num_classes,
                                 dataset=args.dataset, dual_attn_variant=args.dual_attn_variant)
            train_loader, test_loader = build_dataloaders(
                fold["X_train"], y_train_enc, fold["X_test"], y_test_enc,
                batch_size=args.batch_size,
            )
            metrics = train_one_run(model, train_loader, test_loader, device,
                                     epochs=args.epochs, lr=args.lr, patience=args.patience)
            print(f"    -> {metrics}")

            ckpt_path = Path("checkpoints") / args.model / f"{args.dataset}_loso_fold{fold_id}_seed{seed}.pt"
            save_checkpoint(model, ckpt_path)

            append_result_row({
                "model": args.model, "dataset": args.dataset, "split_mode": "loso",
                "fold_id": fold_id, "seed": seed, **metrics, **efficiency,
            })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--dataset", required=True, choices=["ucihar", "wisdm", "pamap2"])
    p.add_argument("--split_mode", required=True, choices=["official", "loso"])
    p.add_argument("--dual_attn_variant", choices=list(DUAL_ATTN_VARIANTS.keys()), default="full",
                   help="ablation arm for dual_attention_bigru only; ignored by other models")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_folds", type=int, default=None, help="override LOSO_N_FOLDS default")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())