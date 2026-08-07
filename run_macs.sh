#!/usr/bin/env bash
# Run the complete workflow from any directory with:
#   bash /path/to/MACS-2/run_macs.sh
#
# This is restartable: an existing non-empty final checkpoint is skipped.
# By default, intermediate single-source epoch checkpoints are removed after
# each final checkpoint is produced to keep the full run within available disk.

set -Eeuo pipefail

# ======================== EDITABLE CONFIGURATION ========================
# Every scalar can also be set as an environment variable. The most useful
# values additionally have command-line options; run `bash run_macs.sh --help`.
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
SINGLE_BATCH_SIZE="${SINGLE_BATCH_SIZE:-8}"
MACS_BATCH_SIZE="${MACS_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ACTIVE_ITERS="${ACTIVE_ITERS:-6}"
TRAIN_EPOCHS_PER_ITER="${TRAIN_EPOCHS_PER_ITER:-5}"
KEEP_EPOCH_CHECKPOINTS="${KEEP_EPOCH_CHECKPOINTS:-0}"
MINIMUM_FREE_GIB="${MINIMUM_FREE_GIB:-9}"
ANNOTATION_BUDGETS_CSV="${ANNOTATION_BUDGETS_CSV:-10,20,40,60,80,100}"
RUN_DOMAINS_CSV="${RUN_DOMAINS_CSV:-}"
RUN_STAGES_CSV="${RUN_STAGES_CSV:-single,macs,eval}"
MODELS_DIR="${MODELS_DIR:-models}"
RESULTS_FILE="${RESULTS_FILE:-multidomain_all_results.csv}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

DOMAINS=(
  c-elegans-dauer-stage
  dhanyasi-P14-mouse-cerebellum
  fly
  katz-lab-berghia-connective
  micron
  octopus-vulgaris-vertical-lobe-glia-deep-neuropil
  octopus-vulgaris-vertical-lobe-sfltract
  snemi
  whole-mouse-brain
)

declare -A SINGLE_EPOCHS=(
  [c-elegans-dauer-stage]=100
  [dhanyasi-P14-mouse-cerebellum]=100
  [fly]=100
  [katz-lab-berghia-connective]=100
  [micron]=100
  [octopus-vulgaris-vertical-lobe-glia-deep-neuropil]=100
  [octopus-vulgaris-vertical-lobe-sfltract]=100
  [snemi]=100
  [whole-mouse-brain]=500
)
# =======================================================================

usage() {
  cat <<'EOF'
Usage: bash run_macs.sh [options]

Stages default to the complete single-source -> MACS -> evaluation pipeline.
Existing non-empty final checkpoints are skipped, so interrupted runs resume.

Options:
  --gpu ID                    Physical GPU exposed as cuda:0 (default: 0)
  --batch-size N              Set all three batch sizes at once
  --single-batch-size N       Single-source training batch size (default: 8)
  --macs-batch-size N         MACS training batch size (default: 4)
  --eval-batch-size N         Evaluation batch size (default: 4)
  --num-workers N             DataLoader worker processes; 0 disables them (default: 2)
  --cpu-threads N             Set both OMP and MKL thread counts
  --omp-num-threads N         OpenMP CPU threads (default: 2)
  --mkl-num-threads N         Intel MKL CPU threads (default: 2)
  --learning-rate RATE        Single-source learning rate (default: 1e-4)
  --active-iters N            Active-learning iterations (default: 6)
  --train-epochs-per-iter N   Fine-tuning epochs per active iteration (default: 5)
  --single-epochs N           Override epochs for every single-source domain
  --whole-mouse-epochs N      Override only whole-mouse-brain epochs
  --budgets LIST              Comma-separated annotation budgets (default: 10,20,40,60,80,100)
  --domains LIST              Comma-separated target domains (default: all nine)
  --stages LIST               Comma-separated stages: single,macs,eval (default: all)
  --models-dir PATH           Checkpoint directory (default: models)
  --results-file PATH         Evaluation CSV path (default: multidomain_all_results.csv)
  --keep-epoch-checkpoints    Retain single-source epoch checkpoints
  --minimum-free-gib N        Required free disk space (default: 9)
  --preflight-only            Validate configuration, dependencies, GPU, data, and disk
  -h, --help                  Show this help

Examples:
  bash run_macs.sh --single-batch-size 2 --macs-batch-size 1
  bash run_macs.sh --stages single --domains fly --single-epochs 1 --models-dir models_smoke
  bash run_macs.sh --stages macs,eval --domains fly --budgets 10
EOF
}

ALL_SINGLE_EPOCHS_OVERRIDE=""
WHOLE_MOUSE_EPOCHS_OVERRIDE=""
while (( $# > 0 )); do
  case "$1" in
    --gpu) GPU_ID="${2:?Missing value for --gpu}"; shift 2 ;;
    --batch-size)
      SINGLE_BATCH_SIZE="${2:?Missing value for --batch-size}"
      MACS_BATCH_SIZE=$SINGLE_BATCH_SIZE
      EVAL_BATCH_SIZE=$SINGLE_BATCH_SIZE
      shift 2
      ;;
    --single-batch-size) SINGLE_BATCH_SIZE="${2:?Missing value for --single-batch-size}"; shift 2 ;;
    --macs-batch-size) MACS_BATCH_SIZE="${2:?Missing value for --macs-batch-size}"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="${2:?Missing value for --eval-batch-size}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:?Missing value for --num-workers}"; shift 2 ;;
    --cpu-threads)
      OMP_NUM_THREADS="${2:?Missing value for --cpu-threads}"
      MKL_NUM_THREADS=$OMP_NUM_THREADS
      shift 2
      ;;
    --omp-num-threads) OMP_NUM_THREADS="${2:?Missing value for --omp-num-threads}"; shift 2 ;;
    --mkl-num-threads) MKL_NUM_THREADS="${2:?Missing value for --mkl-num-threads}"; shift 2 ;;
    --learning-rate) LEARNING_RATE="${2:?Missing value for --learning-rate}"; shift 2 ;;
    --active-iters) ACTIVE_ITERS="${2:?Missing value for --active-iters}"; shift 2 ;;
    --train-epochs-per-iter) TRAIN_EPOCHS_PER_ITER="${2:?Missing value for --train-epochs-per-iter}"; shift 2 ;;
    --single-epochs) ALL_SINGLE_EPOCHS_OVERRIDE="${2:?Missing value for --single-epochs}"; shift 2 ;;
    --whole-mouse-epochs) WHOLE_MOUSE_EPOCHS_OVERRIDE="${2:?Missing value for --whole-mouse-epochs}"; shift 2 ;;
    --budgets) ANNOTATION_BUDGETS_CSV="${2:?Missing value for --budgets}"; shift 2 ;;
    --domains) RUN_DOMAINS_CSV="${2:?Missing value for --domains}"; shift 2 ;;
    --stages) RUN_STAGES_CSV="${2:?Missing value for --stages}"; shift 2 ;;
    --models-dir) MODELS_DIR="${2:?Missing value for --models-dir}"; shift 2 ;;
    --results-file) RESULTS_FILE="${2:?Missing value for --results-file}"; shift 2 ;;
    --keep-epoch-checkpoints) KEEP_EPOCH_CHECKPOINTS=1; shift ;;
    --minimum-free-gib) MINIMUM_FREE_GIB="${2:?Missing value for --minimum-free-gib}"; shift 2 ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS
export MKL_NUM_THREADS

for value_name in SINGLE_BATCH_SIZE MACS_BATCH_SIZE EVAL_BATCH_SIZE OMP_NUM_THREADS MKL_NUM_THREADS ACTIVE_ITERS TRAIN_EPOCHS_PER_ITER MINIMUM_FREE_GIB; do
  value=${!value_name}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$value_name must be a positive integer; got: $value" >&2
    exit 2
  }
done
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || {
  echo "NUM_WORKERS must be a non-negative integer; got: $NUM_WORKERS" >&2
  exit 2
}
[[ "$KEEP_EPOCH_CHECKPOINTS" == 0 || "$KEEP_EPOCH_CHECKPOINTS" == 1 ]] || {
  echo "KEEP_EPOCH_CHECKPOINTS must be 0 or 1; got: $KEEP_EPOCH_CHECKPOINTS" >&2
  exit 2
}
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] || {
  echo "PREFLIGHT_ONLY must be 0 or 1; got: $PREFLIGHT_ONLY" >&2
  exit 2
}
[[ -n "$MODELS_DIR" && -n "$RESULTS_FILE" ]] || {
  echo "MODELS_DIR and RESULTS_FILE cannot be empty" >&2
  exit 2
}

IFS=',' read -r -a BUDGETS <<< "$ANNOTATION_BUDGETS_CSV"
IFS=',' read -r -a RUN_STAGES <<< "$RUN_STAGES_CSV"
if [[ -n "$RUN_DOMAINS_CSV" ]]; then
  IFS=',' read -r -a RUN_DOMAINS <<< "$RUN_DOMAINS_CSV"
else
  RUN_DOMAINS=("${DOMAINS[@]}")
fi

contains() {
  local wanted=$1
  shift
  local item
  for item in "$@"; do
    [[ "$item" == "$wanted" ]] && return 0
  done
  return 1
}

for stage in "${RUN_STAGES[@]}"; do
  contains "$stage" single macs eval || {
    echo "Unknown stage: $stage (expected single, macs, or eval)" >&2
    exit 2
  }
done
for domain in "${RUN_DOMAINS[@]}"; do
  contains "$domain" "${DOMAINS[@]}" || {
    echo "Unknown domain: $domain" >&2
    exit 2
  }
done
for budget in "${BUDGETS[@]}"; do
  [[ "$budget" =~ ^[1-9][0-9]*$ ]] && (( budget <= 100 )) || {
    echo "Each annotation budget must be an integer from 1 to 100; got: $budget" >&2
    exit 2
  }
done

echo "Configuration:"
echo "  GPU_ID=$GPU_ID"
echo "  SINGLE_BATCH_SIZE=$SINGLE_BATCH_SIZE"
echo "  MACS_BATCH_SIZE=$MACS_BATCH_SIZE"
echo "  EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
echo "  NUM_WORKERS=$NUM_WORKERS"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  LEARNING_RATE=$LEARNING_RATE"
echo "  ACTIVE_ITERS=$ACTIVE_ITERS"
echo "  TRAIN_EPOCHS_PER_ITER=$TRAIN_EPOCHS_PER_ITER"
echo "  ANNOTATION_BUDGETS=${BUDGETS[*]}"
echo "  RUN_DOMAINS=${RUN_DOMAINS[*]}"
echo "  RUN_STAGES=${RUN_STAGES[*]}"
echo "  MODELS_DIR=$MODELS_DIR"
echo "  RESULTS_FILE=$RESULTS_FILE"

if [[ -n "$ALL_SINGLE_EPOCHS_OVERRIDE" ]]; then
  [[ "$ALL_SINGLE_EPOCHS_OVERRIDE" =~ ^[1-9][0-9]*$ ]] || {
    echo "--single-epochs must be a positive integer" >&2
    exit 2
  }
  for domain in "${DOMAINS[@]}"; do
    SINGLE_EPOCHS[$domain]=$ALL_SINGLE_EPOCHS_OVERRIDE
  done
fi
if [[ -n "$WHOLE_MOUSE_EPOCHS_OVERRIDE" ]]; then
  [[ "$WHOLE_MOUSE_EPOCHS_OVERRIDE" =~ ^[1-9][0-9]*$ ]] || {
    echo "--whole-mouse-epochs must be a positive integer" >&2
    exit 2
  }
  SINGLE_EPOCHS[whole-mouse-brain]=$WHOLE_MOUSE_EPOCHS_OVERRIDE
fi

# Download and extract the dataset only when ./data is not already present.
if [[ ! -d data ]]; then
  command -v gdown >/dev/null 2>&1 || {
    echo "gdown is required. Install it with: python -m pip install gdown" >&2
    exit 1
  }
  command -v unzip >/dev/null 2>&1 || {
    echo "unzip is required to extract data.zip" >&2
    exit 1
  }
  gdown 1-tuV1zWgcsaz9tEUDRpsrZPoEj9e_glZ -O data.zip
  unzip data.zip
  [[ -d data ]] || {
    echo "The archive did not create the expected $REPO_DIR/data directory." >&2
    exit 1
  }
  rm -f data.zip
fi

# Environment and dataset preflight checks.
"$PYTHON_BIN" - <<'PY'
import importlib
import torch

required = ("torch", "torchvision", "PIL", "numpy", "tqdm", "sklearn", "scipy")
for module_name in required:
    importlib.import_module(module_name)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch. Run this outside the sandbox on the GPU host.")

print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

for domain in "${DOMAINS[@]}"; do
  for split in train val test; do
    for kind in raw membranes; do
      path="data/$domain/$split/$kind"
      [[ -d "$path" ]] || {
        echo "Missing dataset directory: $path" >&2
        exit 1
      }
      compgen -G "$path/*" >/dev/null || {
        echo "Dataset directory is empty: $path" >&2
        exit 1
      }
    done
  done
  mkdir -p "$MODELS_DIR/$domain"
done

if [[ "$KEEP_EPOCH_CHECKPOINTS" == 1 ]]; then
  minimum_free_gib=$MINIMUM_FREE_GIB
  (( minimum_free_gib < 24 )) && minimum_free_gib=24
else
  minimum_free_gib=$MINIMUM_FREE_GIB
fi
minimum_free_kib=$((minimum_free_gib * 1024 * 1024))
available_kib=$(df -Pk "$REPO_DIR" | awk 'NR == 2 {print $4}')
if (( available_kib < minimum_free_kib )); then
  echo "Insufficient free disk space for the requested workflow." >&2
  echo "Available: $((available_kib / 1024 / 1024)) GiB; required: at least $minimum_free_gib GiB." >&2
  exit 1
fi
df -h "$REPO_DIR"

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo "Preflight complete; no training or evaluation was started."
  exit 0
fi

# Stage 1: train the selected single-source models sequentially.
if contains single "${RUN_STAGES[@]}"; then
  for domain in "${RUN_DOMAINS[@]}"; do
    final_checkpoint="$MODELS_DIR/$domain/model_final.pth"
    if [[ -s "$final_checkpoint" ]]; then
      echo "[single] Skipping $domain; found $final_checkpoint"
    else
      echo "[single] Training $domain for ${SINGLE_EPOCHS[$domain]} epochs"
      "$PYTHON_BIN" train_single.py \
        --data_root data \
        --domain "$domain" \
        --out_path "$MODELS_DIR/$domain/model.pth" \
        --batch_size "$SINGLE_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --epochs "${SINGLE_EPOCHS[$domain]}" \
        --lr "$LEARNING_RATE" \
        --device cuda:0
      [[ -s "$final_checkpoint" ]] || {
        echo "Missing final source checkpoint: $final_checkpoint" >&2
        exit 1
      }
    fi

    if [[ "$KEEP_EPOCH_CHECKPOINTS" != 1 ]]; then
      find "$MODELS_DIR/$domain" -maxdepth 1 -type f -name 'model_epoch*.pth' -delete
    fi
  done
fi

# Stage 2: train MACS for the selected targets and annotation budgets.
if contains macs "${RUN_STAGES[@]}"; then
  # Every target uses the other eight domains as sources.
  for domain in "${DOMAINS[@]}"; do
    [[ -s "$MODELS_DIR/$domain/model_final.pth" ]] || {
      echo "Missing source checkpoint required by MACS: $MODELS_DIR/$domain/model_final.pth" >&2
      exit 1
    }
  done

  for budget in "${BUDGETS[@]}"; do
    for target in "${RUN_DOMAINS[@]}"; do
      output="$MODELS_DIR/$target/multidomain_final_$budget.pth"
      if [[ -s "$output" ]]; then
        echo "[MACS] Skipping target=$target budget=$budget; found $output"
        continue
      fi

      sources=()
      source_checkpoints=()
      for source in "${DOMAINS[@]}"; do
        if [[ "$source" != "$target" ]]; then
          sources+=("$source")
          source_checkpoints+=("$MODELS_DIR/$source/model_final.pth")
        fi
      done

      echo "[MACS] Training target=$target budget=$budget"
      "$PYTHON_BIN" multidomain.py \
        --data_root data \
        --source_domains "${sources[@]}" \
        --source_ckpts "${source_checkpoints[@]}" \
        --target_domain "$target" \
        --out_path "$output" \
        --active_iters "$ACTIVE_ITERS" \
        --annot_budget "$budget" \
        --train_epochs_per_iter "$TRAIN_EPOCHS_PER_ITER" \
        --batch_size "$MACS_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --device cuda:0
      [[ -s "$output" ]] || {
        echo "Missing final MACS checkpoint: $output" >&2
        exit 1
      }
    done
  done
fi

# Stage 3: evaluate the selected MACS models.
if contains eval "${RUN_STAGES[@]}"; then
  echo "domain,percent,dice,iou,f1,recall,vi" > "$RESULTS_FILE"
  for domain in "${RUN_DOMAINS[@]}"; do
    for budget in "${BUDGETS[@]}"; do
      model_path="$MODELS_DIR/$domain/multidomain_final_$budget.pth"
      [[ -s "$model_path" ]] || {
        echo "Missing MACS checkpoint required by evaluation: $model_path" >&2
        exit 1
      }
      "$PYTHON_BIN" test.py \
        --data_root data \
        --domain "$domain" \
        --model_path "$model_path" \
        --batch_size "$EVAL_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --device cuda:0 >> "$RESULTS_FILE"
    done
  done
fi

echo "Complete: requested stages finished successfully."
if contains single "${RUN_STAGES[@]}" || contains macs "${RUN_STAGES[@]}"; then
  echo "Checkpoints are under $MODELS_DIR"
fi
if contains eval "${RUN_STAGES[@]}"; then
  echo "Evaluation metrics are in $RESULTS_FILE"
fi
