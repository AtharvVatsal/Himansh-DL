"""
data/download_data.py

Downloads and extracts the raw public datasets into the exact layout
src/preprocessing.py expects:

    data/raw/ucihar/UCI HAR Dataset/...
    data/raw/wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt
    data/raw/pamap2/PAMAP2_Dataset/Protocol/subject10X.dat

(preprocessing.py's loaders auto-detect the nested folder each archive
extracts with, so no manual re-shuffling of files is needed after this runs.)

Usage:
    python3 data/download_data.py --dataset all
    python3 data/download_data.py --dataset ucihar
    python3 data/download_data.py --dataset wisdm
    python3 data/download_data.py --dataset pamap2

Only the standard library is used (urllib, zipfile, tarfile) so this runs
with no extra pip installs on a fresh pod.
"""

import argparse
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"

SOURCES = {
    "ucihar": dict(
        url="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        archive_name="ucihar.zip",
        dest="ucihar",
        check_path="ucihar/UCI HAR Dataset/train/Inertial Signals",
    ),
    "wisdm": dict(
        url="https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz",
        archive_name="wisdm.tar.gz",
        dest="wisdm",
        check_path="wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt",
    ),
    "pamap2": dict(
        url="https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
        archive_name="pamap2.zip",
        dest="pamap2",
        check_path="pamap2/PAMAP2_Dataset/Protocol",
    ),
}


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    done = block_num * block_size
    pct = min(100, done * 100 // total_size)
    sys.stdout.write(f"\r  downloading... {pct:3d}% ({done // (1 << 20)}MB / {total_size // (1 << 20)}MB)")
    sys.stdout.flush()


def _download(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{dest_path.parent.name}] GET {url}")
    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print()  # newline after progress bar


def _extract_one(archive_path: Path, extract_dir: Path):
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
    else:  # .tar.gz
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_dir)


def _extract(archive_path: Path, extract_dir: Path, keep_archive: bool = False):
    """Extract archive_path into extract_dir, then repeatedly extract any
    .zip files that appear inside (the UCI ML repo's newer "static" download
    links wrap the real dataset zip -- e.g. UCI-HAR's static archive contains
    'UCI HAR Dataset.zip' rather than an already-extracted folder -- so one
    level of extraction isn't always enough)."""
    print(f"[{extract_dir.name}] extracting {archive_path.name} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    _extract_one(archive_path, extract_dir)

    for _ in range(3):  # bounded, in case of multiple nesting levels
        nested_zips = list(extract_dir.rglob("*.zip"))
        if not nested_zips:
            break
        for nested in nested_zips:
            print(f"[{extract_dir.name}] found nested archive {nested.name}, extracting ...")
            _extract_one(nested, nested.parent)
            if not keep_archive:
                nested.unlink()


def fetch(dataset: str, keep_archive: bool = False):
    cfg = SOURCES[dataset]
    dest_dir = RAW_DIR / cfg["dest"]
    check_path = RAW_DIR / cfg["check_path"]

    if check_path.exists():
        print(f"[{dataset}] already present at {check_path}, skipping. "
              f"(delete it and re-run to force a fresh download)")
        return

    archive_path = RAW_DIR / cfg["archive_name"]
    _download(cfg["url"], archive_path)
    _extract(archive_path, dest_dir, keep_archive=keep_archive)

    if not keep_archive:
        archive_path.unlink()

    if not check_path.exists():
        raise RuntimeError(
            f"[{dataset}] extraction finished but expected path {check_path} "
            f"not found -- the upstream archive layout may have changed; "
            f"inspect {dest_dir} manually."
        )
    print(f"[{dataset}] ready at {check_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["ucihar", "wisdm", "pamap2", "all"], default="all")
    p.add_argument("--keep_archive", action="store_true",
                   help="keep the downloaded .zip/.tar.gz after extracting (default: delete to save disk)")
    args = p.parse_args()

    datasets = list(SOURCES.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        fetch(ds, keep_archive=args.keep_archive)

    free_gb = shutil.disk_usage(RAW_DIR).free / (1 << 30)
    print(f"\nDone. {free_gb:.1f}GB free on this volume.")


if __name__ == "__main__":
    main()