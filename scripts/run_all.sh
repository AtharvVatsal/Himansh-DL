#!/usr/bin/env bash
# scripts/run_all.sh -- the complete pipeline, end to end.
#
#   bash scripts/run_all.sh                 # everything, datasets one after another
#   PARALLEL=1 bash scripts/run_all.sh      # train the 3 datasets concurrently (1 GPU is enough)
#   DATASETS="ucihar" bash scripts/run_all.sh
#   STEPS="train evaluate" bash scripts/run_all.sh   # run only some steps
#
# Steps: download preprocess smoke train profile stats evaluate
# Safe to re-run: downloads are skipped if present and every finished training
# run is skipped (resume from results/tables/raw_results.csv).
# Logs: results/logs/<step or dataset>.log
#
# Environment: on first run a virtualenv is created in .venv and
# requirements.txt is installed into it (re-installed whenever
# requirements.txt changes). If an NVIDIA GPU is present (nvidia-smi on PATH),
# the CUDA build of PyTorch is installed and the run aborts if CUDA is not
# usable, instead of silently training on the CPU.
#   VENV=.venv            where to put the virtualenv
#   BASE_PY=python3.12    interpreter used to create it (default: first working of py -3 / python3 / python)
#   TORCH_CUDA=cu128      PyTorch CUDA wheel index (https://download.pytorch.org/whl/<TORCH_CUDA>)
#   NO_VENV=1 PY=python   skip venv management and use this interpreter as-is
# Windows: run from Git Bash (bash + tee are needed), not PowerShell/cmd.

set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS=${DATASETS:-"ucihar wisdm pamap2"}
STEPS=${STEPS:-"download preprocess smoke train profile stats evaluate"}
PARALLEL=${PARALLEL:-0}
DEVICE=${DEVICE:-auto}
VENV=${VENV:-.venv}
TORCH_CUDA=${TORCH_CUDA:-cu128}
MODELS="bigru_baseline bilstm_attention_baseline cnn_bigru_reference dual_attention_bigru"
mkdir -p results/logs

# UTF-8 file/console I/O on Windows (default there is cp1252); unbuffered so
# `| tee` logs show progress live instead of in 8KB chunks
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

has_step() { [[ " $STEPS " == *" $1 "* ]]; }
banner()   { echo; echo "==================== $* ===================="; }

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
# a usable interpreter >= 3.10 (skips the Windows Store "python" stub, which
# exists on PATH but fails when run)
find_base_python() {
  local c
  for c in ${BASE_PY:-} "py -3" python3 python; do
    if $c -c "import sys; sys.exit(sys.version_info < (3, 10))" >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  return 1
}

setup_env() {
  banner "0/7 python environment ($VENV)"
  if [[ ! -d $VENV ]]; then
    local base
    base=$(find_base_python) || { echo "no Python >= 3.10 found; install one or set BASE_PY=..."; exit 1; }
    echo "creating $VENV with: $base ($($base --version 2>&1))"
    $base -m venv "$VENV"
  fi
  if [[ -x $VENV/Scripts/python.exe ]]; then PY="$VENV/Scripts/python.exe"   # Windows
  else PY="$VENV/bin/python"; fi

  local has_gpu=0
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && has_gpu=1

  # (re)install only when requirements.txt changed since the last install
  local stamp=$VENV/.installed_requirements.txt
  if ! cmp -s requirements.txt "$stamp"; then
    $PY -m pip install --upgrade pip
    if [[ $has_gpu == 1 ]]; then
      echo "NVIDIA GPU found -> installing CUDA PyTorch from the $TORCH_CUDA index"
      $PY -m pip install torch --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
    fi
    $PY -m pip install -r requirements.txt
    cp requirements.txt "$stamp"
  else
    echo "requirements already installed (delete $stamp to force a reinstall)"
  fi

  $PY -c "import torch, sys; print(f'python {sys.version.split()[0]} | torch {torch.__version__} | cuda available: {torch.cuda.is_available()}' + (f' ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ''))"
  if [[ $has_gpu == 1 ]] && ! $PY -c "import torch, sys; sys.exit(not torch.cuda.is_available())"; then
    echo "An NVIDIA GPU is present but this PyTorch cannot use it (CPU-only build or driver too old)."
    echo "Fix: rm $stamp && $PY -m pip uninstall -y torch, update the NVIDIA driver, then re-run"
    echo "     (or try another wheel index, e.g. TORCH_CUDA=cu126). To train on CPU anyway: DEVICE=cpu."
    [[ $DEVICE == cpu ]] || exit 1
  fi
}

if [[ ${NO_VENV:-0} == 1 ]]; then
  PY=${PY:-$(find_base_python)} || { echo "no Python >= 3.10 found; set PY=..."; exit 1; }
else
  setup_env 2>&1 | tee results/logs/setup.log
  # the pipe runs setup_env in a subshell, so resolve PY again here
  if [[ -x $VENV/Scripts/python.exe ]]; then PY="$VENV/Scripts/python.exe"; else PY="$VENV/bin/python"; fi
fi

train_dataset() {
  local ds=$1
  for m in $MODELS; do
    local extra=""
    [[ $m == dual_attention_bigru ]] && extra="--dual_attn_variant all"
    $PY -m src.train --config "configs/${ds}_config.yaml" --model "$m" --split_mode both \
        --device "$DEVICE" $extra
  done
}

if has_step download; then
  banner "1/7 download raw datasets"
  $PY data/download_data.py --dataset all 2>&1 | tee results/logs/download.log
fi

if has_step preprocess; then
  banner "2/7 preprocess (window + save .npy)"
  $PY -m src.preprocessing --dataset all 2>&1 | tee results/logs/preprocess.log
  $PY -m src.data_loader 2>&1 | tee -a results/logs/preprocess.log
fi

if has_step smoke; then
  banner "3/7 smoke test (1 epoch, 1 fold; writes results/smoke/ only)"
  for ds in $DATASETS; do
    for m in $MODELS; do
      $PY -m src.train --config "configs/${ds}_config.yaml" --model "$m" --split_mode both \
          --dual_attn_variant all --device "$DEVICE" --smoke
    done
  done 2>&1 | tee results/logs/smoke.log
fi

if has_step train; then
  banner "4/7 train all models x variants x folds x seeds"
  if [[ $PARALLEL == 1 ]]; then
    pids=()
    for ds in $DATASETS; do
      ( train_dataset "$ds" > "results/logs/train_${ds}.log" 2>&1 ) &
      pids+=($!)
      echo "started $ds (pid $!) -> results/logs/train_${ds}.log"
    done
    fail=0
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    [[ $fail == 0 ]] || { echo "a training job failed -- see results/logs/train_*.log; re-run to resume"; exit 1; }
  else
    for ds in $DATASETS; do
      train_dataset "$ds" 2>&1 | tee "results/logs/train_${ds}.log"
    done
  fi
fi

if has_step profile; then
  banner "5/7 efficiency profiling (params / FLOPs / latency, one controlled pass)"
  $PY -m src.profile_efficiency --datasets $DATASETS 2>&1 | tee results/logs/profile.log
fi

if has_step stats; then
  banner "6/7 significance tests (Wilcoxon / paired t, LOSO folds)"
  $PY -m src.stats_test 2>&1 | tee results/logs/stats.log
fi

if has_step evaluate; then
  banner "7/7 master table, confusion matrices, figures"
  $PY -m src.evaluate 2>&1 | tee results/logs/evaluate.log
fi

banner "done"
echo "results/tables/raw_results.csv       every run (one row per model/variant/dataset/split/fold/seed)"
echo "results/tables/master_comparison.md  paper table   (.csv alongside)"
echo "results/tables/significance.csv      p-values      results/tables/efficiency.csv  latency"
echo "results/figures/                     plots         results/preds/                 per-run predictions"
