"""
src/preprocessing.py

Loads raw sensor data for UCI-HAR, WISDM, and PAMAP2, and converts each into
windowed (X, y, subject_id) arrays saved to data/processed/<dataset>/.

ASSUMED RAW LAYOUT (standard public distributions — adjust paths below if
your download_data.py produces something different):

  data/raw/ucihar/
      train/Inertial Signals/{signal}_train.txt   (9 files, 128-sample rows)
      test/Inertial Signals/{signal}_test.txt
      train/y_train.txt, train/subject_train.txt
      test/y_test.txt,   test/subject_test.txt
    -> UCI-HAR ships PRE-WINDOWED (128 samples @ 50Hz = 2.56s, 50% overlap).
       We do not re-window it; we just assemble channels + labels + subjects.

  data/raw/wisdm/WISDM_ar_v1.1_raw.txt
    -> single file, lines of the form:
       user,activity,timestamp,x,y,z;
       Continuous accelerometer-only stream. We window it ourselves.

  data/raw/pamap2/Protocol/subject10X.dat  (X = 1..9)
    -> space-delimited, 54 columns per row:
       col 0: timestamp, col 1: activityID, col 2: heart rate,
       col 3-19: hand IMU (temp, acc16 x/y/z, acc6 x/y/z, gyro x/y/z,
                            mag x/y/z, orientation x/y/z/w — 17 cols),
       col 20-36: chest IMU (same layout),
       col 37-53: ankle IMU (same layout).
       activityID == 0 marks transient/unlabeled rows and is dropped.

NATIVE MODE (default, used for the core 4x3 comparison table):
  each dataset keeps its own sampling rate / window size / channel set.

HARMONIZED MODE (optional, cross-dataset transfer experiment only):
  20 Hz common rate (downsample-only), 2.56s windows @ 50% overlap,
  triaxial accelerometer channels only. See docs/research_gap_notes.md §3
  for the full rationale, including the documented (not "fixed") sensor
  placement confound.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample_poly
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# Per-dataset native configs
# ---------------------------------------------------------------------------
NATIVE_CONFIGS = {
    "ucihar": dict(sampling_rate=50, window_samples=128, step_samples=64),  # pre-windowed by source
    "wisdm":  dict(sampling_rate=20, window_samples=51, step_samples=25),   # ~2.56s @ 20Hz, 50% overlap
    "pamap2": dict(sampling_rate=100, window_samples=256, step_samples=128),  # 2.56s @ 100Hz, 50% overlap
}

HARMONIZED_CONFIG = dict(
    sampling_rate=20,
    window_seconds=2.56,
    window_samples=51,   # round(2.56 * 20)
    overlap=0.5,
    channels=["acc_x", "acc_y", "acc_z"],
)

PAMAP2_HARMONIZED_PLACEMENT = "hand"  # closest analog to WISDM's wrist/pocket placement


# ---------------------------------------------------------------------------
# Generic windowing utility (for continuous streams: WISDM, PAMAP2)
# ---------------------------------------------------------------------------
def sliding_windows(signal: np.ndarray, labels: np.ndarray, subjects: np.ndarray,
                     window_size: int, step: int):
    """
    Slice a continuous (T, C) signal into overlapping windows.
    A window's label is the majority label within it; windows that span a
    subject or activity boundary are dropped (label / subject not constant).

    Returns X (N, window_size, C), y (N,), subj (N,)
    """
    X, y, subj = [], [], []
    n = signal.shape[0]
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        lbl_slice = labels[start:end]
        subj_slice = subjects[start:end]
        if len(np.unique(lbl_slice)) != 1 or len(np.unique(subj_slice)) != 1:
            continue  # boundary window, discard
        X.append(signal[start:end])
        y.append(lbl_slice[0])
        subj.append(subj_slice[0])
    return np.array(X), np.array(y), np.array(subj)


def resample_signal(x: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """
    Downsample-only resampling with anti-aliasing (scipy resample_poly).
    x: (T, C) or (T,). Raises if target_rate > orig_rate (upsampling is
    disallowed per the harmonization decision — see research_gap_notes.md).
    """
    if target_rate > orig_rate:
        raise ValueError(
            f"Upsampling not allowed ({orig_rate}Hz -> {target_rate}Hz). "
            "Harmonized mode is downsample-only by design."
        )
    from math import gcd
    g = gcd(orig_rate, target_rate)
    up, down = target_rate // g, orig_rate // g
    axis = 0
    return resample_poly(x, up, down, axis=axis)


def normalize_channels(X_train: np.ndarray, *others: np.ndarray):
    """
    Fit per-channel mean/std on X_train ONLY (shape N, T, C), apply to
    X_train and any number of other arrays with the same channel dim.
    Returns (X_train_norm, *others_norm, mean, std).
    Call this per-split (official split or per LOSO fold) -- never on
    the full dataset before splitting, to avoid test-fold leakage.
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    out = [(X_train - mean) / std]
    for arr in others:
        out.append((arr - mean) / std)
    return (*out, mean.squeeze(), std.squeeze())


# ---------------------------------------------------------------------------
# UCI-HAR
# ---------------------------------------------------------------------------
UCIHAR_SIGNALS = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]  # 6 channels: raw total acceleration + gyro. (body_acc is derivable from
   # total_acc minus gravity and is intentionally omitted to avoid redundant
   # channels -- adjust here if you want all 9 original UCI-HAR signals.)


def load_raw_ucihar(raw_dir: Path = RAW_DIR / "ucihar") -> dict:
    """Load UCI-HAR's pre-windowed official train/test split.

    The official zip extracts with an extra 'UCI HAR Dataset/' folder in
    between (i.e. data/raw/ucihar/UCI HAR Dataset/train/...) -- auto-detect
    that layout so this works whether or not that extra folder is present.
    """
    if (raw_dir / "UCI HAR Dataset").exists():
        raw_dir = raw_dir / "UCI HAR Dataset"

    def _load_split(split: str):
        base = raw_dir / split / "Inertial Signals"
        chans = []
        for sig in UCIHAR_SIGNALS:
            fpath = base / f"{sig}_{split}.txt"
            chans.append(np.loadtxt(fpath))  # (N, 128)
        X = np.stack(chans, axis=-1)  # (N, 128, 6)
        y = np.loadtxt(raw_dir / split / f"y_{split}.txt").astype(int)
        subj = np.loadtxt(raw_dir / split / f"subject_{split}.txt").astype(int)
        return X, y, subj

    X_train, y_train, subj_train = _load_split("train")
    X_test, y_test, subj_test = _load_split("test")

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    subj = np.concatenate([subj_train, subj_test])
    # keep the official split mask so data_loader.py can reproduce it exactly
    official_split = np.array(["train"] * len(y_train) + ["test"] * len(y_test))
    return dict(X=X, y=y, subject=subj, official_split=official_split)


# ---------------------------------------------------------------------------
# WISDM
# ---------------------------------------------------------------------------
WISDM_ACTIVITY_MAP = {
    "Walking": 0, "Jogging": 1, "Upstairs": 2,
    "Downstairs": 3, "Sitting": 4, "Standing": 5,
}


def load_raw_wisdm(raw_path: Path = RAW_DIR / "wisdm" / "WISDM_ar_v1.1_raw.txt",
                    cfg: dict = NATIVE_CONFIGS["wisdm"]) -> dict:
    """Parse WISDM's semicolon-terminated CSV-like raw file and window it.

    The official tar.gz extracts with an extra 'WISDM_ar_v1.1/' folder in
    between (i.e. data/raw/wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt) --
    auto-detect that layout so this works whether or not that folder exists.
    """
    if not raw_path.exists():
        nested = raw_path.parent / "WISDM_ar_v1.1" / raw_path.name
        if nested.exists():
            raw_path = nested
        else:
            raise FileNotFoundError(
                f"Could not find WISDM raw file at {raw_path} or {nested}. "
                f"Run `python data/download_data.py --dataset wisdm` first."
            )

    rows = []
    with open(raw_path, "r") as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 6:
                continue  # malformed line, skip
            try:
                user, activity, ts, x, y, z = parts
                rows.append((int(user), WISDM_ACTIVITY_MAP[activity],
                             float(x), float(y), float(z)))
            except (ValueError, KeyError):
                continue  # bad row (NaN fields, unknown activity), skip

    df = pd.DataFrame(rows, columns=["user", "activity", "x", "y", "z"])
    df = df.sort_values(["user", "activity"]).reset_index(drop=True)

    signal = df[["x", "y", "z"]].to_numpy()
    labels = df["activity"].to_numpy()
    subjects = df["user"].to_numpy()

    X, y, subj = sliding_windows(signal, labels, subjects,
                                  cfg["window_samples"], cfg["step_samples"])
    return dict(X=X, y=y, subject=subj, official_split=None)


# ---------------------------------------------------------------------------
# PAMAP2
# ---------------------------------------------------------------------------
PAMAP2_HAND_ACC_COLS = [4, 5, 6]    # acc16 x,y,z within hand IMU block (0-indexed, see header note)
PAMAP2_HAND_GYRO_COLS = [10, 11, 12]
PAMAP2_ACTIVITY_COL = 1


def _resolve_pamap2_protocol_dir(raw_dir: Path) -> Path:
    """
    The official zip extracts with an extra 'PAMAP2_Dataset/' folder in
    between (i.e. data/raw/pamap2/PAMAP2_Dataset/Protocol/subjectXXX.dat) --
    auto-detect that layout so callers work whether or not that folder
    exists. Shared by load_raw_pamap2 and harmonize_pamap2.
    """
    if raw_dir.exists():
        return raw_dir
    nested = raw_dir.parent / "PAMAP2_Dataset" / "Protocol"
    if nested.exists():
        return nested
    raise FileNotFoundError(
        f"Could not find PAMAP2 Protocol folder at {raw_dir} or {nested}. "
        f"Run `python data/download_data.py --dataset pamap2` first."
    )


def load_raw_pamap2(raw_dir: Path = RAW_DIR / "pamap2" / "Protocol",
                     cfg: dict = NATIVE_CONFIGS["pamap2"]) -> dict:
    """
    Load all subject .dat files, drop transient rows (activityID==0),
    forward-fill occasional NaNs (sensor dropout), window per subject.
    Native mode uses hand+chest+ankle acc+gyro (12 channels); adjust
    PAMAP2_*_COLS below if you want a different channel subset.
    """
    raw_dir = _resolve_pamap2_protocol_dir(raw_dir)

    all_X, all_y, all_subj = [], [], []
    for fpath in sorted(raw_dir.glob("subject*.dat")):
        subject_id = int(fpath.stem.replace("subject", ""))
        data = pd.read_csv(fpath, sep=r"\s+", header=None)
        data = data[data[PAMAP2_ACTIVITY_COL] != 0]  # drop transient/unlabeled
        data = data.ffill().bfill()  # handle sensor dropout NaNs

        # Native mode: hand accel + gyro only, for simplicity/consistency
        # across models. Extend to chest/ankle columns here if desired.
        signal = data.iloc[:, PAMAP2_HAND_ACC_COLS + PAMAP2_HAND_GYRO_COLS].to_numpy()
        labels = data[PAMAP2_ACTIVITY_COL].to_numpy().astype(int)
        subjects = np.full(len(labels), subject_id)

        X, y, subj = sliding_windows(signal, labels, subjects,
                                      cfg["window_samples"], cfg["step_samples"])
        all_X.append(X)
        all_y.append(y)
        all_subj.append(subj)

    return dict(
        X=np.concatenate(all_X), y=np.concatenate(all_y),
        subject=np.concatenate(all_subj), official_split=None,
    )


# ---------------------------------------------------------------------------
# Harmonized-mode conversion (optional bonus experiment only)
# ---------------------------------------------------------------------------
def harmonize_ucihar(native: dict) -> dict:
    """UCI-HAR windows are already 2.56s; just resample 128->51 samples and
    keep only the accel channels (total_acc_x/y/z, i.e. channels 0:3)."""
    X = native["X"][:, :, 0:3]  # total_acc x,y,z
    X_h = resample_poly(X, HARMONIZED_CONFIG["window_samples"], X.shape[1], axis=1)
    return dict(X=X_h, y=native["y"], subject=native["subject"])


def harmonize_wisdm(raw_signal: np.ndarray, labels: np.ndarray, subjects: np.ndarray) -> dict:
    """WISDM is already accel-only and already ~20Hz; just re-window at the
    harmonized window/step size (no resampling needed)."""
    X, y, subj = sliding_windows(
        raw_signal, labels, subjects,
        HARMONIZED_CONFIG["window_samples"],
        int(HARMONIZED_CONFIG["window_samples"] * (1 - HARMONIZED_CONFIG["overlap"])),
    )
    return dict(X=X, y=y, subject=subj)


def harmonize_pamap2(raw_dir: Path = RAW_DIR / "pamap2" / "Protocol") -> dict:
    """Use only the hand IMU accel stream (documented placement choice),
    downsample 100Hz -> 20Hz, then window at the harmonized size."""
    raw_dir = _resolve_pamap2_protocol_dir(raw_dir)

    all_X, all_y, all_subj = [], [], []
    for fpath in sorted(raw_dir.glob("subject*.dat")):
        subject_id = int(fpath.stem.replace("subject", ""))
        data = pd.read_csv(fpath, sep=r"\s+", header=None)
        data = data[data[PAMAP2_ACTIVITY_COL] != 0]
        data = data.ffill().bfill()

        acc = data.iloc[:, PAMAP2_HAND_ACC_COLS].to_numpy()
        labels = data[PAMAP2_ACTIVITY_COL].to_numpy().astype(int)

        acc_ds = resample_signal(acc, orig_rate=100, target_rate=HARMONIZED_CONFIG["sampling_rate"])
        # labels/subjects must be downsampled by the same integer factor for
        # sliding_windows' boundary check to stay meaningful
        factor = len(labels) / len(acc_ds)
        idx = (np.arange(len(acc_ds)) * factor).astype(int).clip(max=len(labels) - 1)
        labels_ds = labels[idx]
        subj_ds = np.full(len(acc_ds), subject_id)

        X, y, subj = sliding_windows(
            acc_ds, labels_ds, subj_ds,
            HARMONIZED_CONFIG["window_samples"],
            int(HARMONIZED_CONFIG["window_samples"] * (1 - HARMONIZED_CONFIG["overlap"])),
        )
        all_X.append(X); all_y.append(y); all_subj.append(subj)

    return dict(X=np.concatenate(all_X), y=np.concatenate(all_y),
                subject=np.concatenate(all_subj))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def process_and_save(dataset: str, mode: str = "native"):
    assert dataset in ("ucihar", "wisdm", "pamap2")
    assert mode in ("native", "harmonized")

    out_dir = PROCESSED_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "native":
        loader = {"ucihar": load_raw_ucihar, "wisdm": load_raw_wisdm, "pamap2": load_raw_pamap2}[dataset]
        result = loader()
        suffix = ""
    else:
        if dataset == "ucihar":
            result = harmonize_ucihar(load_raw_ucihar())
        elif dataset == "pamap2":
            result = harmonize_pamap2()
        else:  # wisdm -- need raw signal/labels/subjects before windowing
            raise NotImplementedError(
                "harmonize_wisdm expects pre-parsed (signal, labels, subjects); "
                "call load_raw_wisdm's internals directly if you need this path."
            )
        suffix = "_harmonized"

    np.save(out_dir / f"X{suffix}.npy", result["X"])
    np.save(out_dir / f"y{suffix}.npy", result["y"])
    np.save(out_dir / f"subject{suffix}.npy", result["subject"])
    if result.get("official_split") is not None:
        np.save(out_dir / f"official_split{suffix}.npy", result["official_split"])

    print(f"[{dataset}/{mode}] X={result['X'].shape} y={result['y'].shape} "
          f"subjects={len(np.unique(result['subject']))} unique")


if __name__ == "__main__":
    # Native mode is the submission-critical path -- run this first for all 3.
    for ds in ("ucihar", "wisdm", "pamap2"):
        process_and_save(ds, mode="native")

    # Harmonized mode: only needed for the optional Week-10 bonus experiment.
    # Uncomment once native-mode results for all 3 models/datasets are done.
    # for ds in ("ucihar", "pamap2"):
    #     process_and_save(ds, mode="harmonized")