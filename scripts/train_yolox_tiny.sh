#!/usr/bin/env bash
set -euo pipefail

# Activate conda environment if available
if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate mmdet 2>/dev/null || true
fi

DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$DIR/common.sh"

# 使用例:
#   bash scripts/train_yolox_tiny.sh                 # 既定(1epoch, AMP有効)
#   EPOCHS=10 bash scripts/train_yolox_tiny.sh       # 10 epoch 学習
#   WORKDIR=work_dirs/exp1 bash scripts/train_yolox_tiny.sh
#   EXTRA_CFG_OPTIONS="default_hooks.logger.interval=20" bash scripts/train_yolox_tiny.sh

export STAGE=train
EPOCHS=${EPOCHS:-1}

# 出力ディレクトリ構築
RUN_OUTPUT_DIR="${OUT_ROOT}/${STAGE}/${DATASET}/${MODEL}/${CONFIG_BASE}/${TAG}/${RUN_ID}"
export RUN_OUTPUT_DIR

# MMEngineのwork_dirとして利用
WORKDIR="${RUN_OUTPUT_DIR}"
mkdir -p "$WORKDIR"

if [[ -z "${USE_AMP:-}" ]]; then
  if [[ "${DEVICE:-}" == "cuda" ]]; then USE_AMP=1; else USE_AMP=0; fi
fi
AMP_FLAG=""
if [[ "${USE_AMP}" == "1" ]]; then AMP_FLAG="--amp"; fi

# Handle IMG_SCALE environment variable for resolution override
if [[ -n "${IMG_SCALE:-}" ]]; then
  # Adjust batch size based on resolution
  if [[ "${IMG_SCALE}" -ge 3000 ]]; then
    BATCH_SIZE=1
  elif [[ "${IMG_SCALE}" -ge 2048 ]]; then
    BATCH_SIZE=2
  elif [[ "${IMG_SCALE}" -ge 1280 ]]; then
    BATCH_SIZE=4
  else
    BATCH_SIZE=8
  fi
  
  # Set batch sizes only - resolution will be set in config file
  SCALE_OPTIONS="train_dataloader.batch_size=${BATCH_SIZE} "
  SCALE_OPTIONS+="val_dataloader.batch_size=${BATCH_SIZE} "
  
  EXTRA_CFG_OPTIONS="${SCALE_OPTIONS} ${EXTRA_CFG_OPTIONS:-}"
fi

python tools/train.py "$CONFIG" \
  ${AMP_FLAG} \
  --work-dir "$WORKDIR" \
  --clearml \
  --clearml-project mmdetection \
  --cfg-options \
    train_cfg.max_epochs="$EPOCHS" \
    default_hooks.checkpoint.interval=1 \
    ${EXTRA_CFG_OPTIONS:-}

# --- Run log ---
LOG_FILE=".notes/py/RUN_LOG.md"
mkdir -p .notes/py
{
  printf '## %s train\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf -- '- CONFIG: %s\n' "$CONFIG"
  printf -- '- WEIGHTS: %s\n' "$WEIGHTS"
  printf -- '- EPOCHS: %s\n' "${EPOCHS:-}"
  printf -- '- DEVICE: %s  USE_AMP: %s\n' "${DEVICE:-}" "${USE_AMP:-0}"
  printf -- '- DATASET: %s  MODEL: %s  TAG: %s\n' "${DATASET:-}" "${MODEL:-}" "${TAG:-}"
  printf -- '- RUN_OUTPUT_DIR: %s\n' "${RUN_OUTPUT_DIR:-}"
  printf -- '- EXTRA_CFG_OPTIONS: %s\n' "${EXTRA_CFG_OPTIONS:-}"
  printf -- '- Command:\n'
  printf '  python tools/train.py "%s" \\\n' "$CONFIG"
  if [[ -n "$AMP_FLAG" ]]; then printf '    %s \\\n' "$AMP_FLAG"; fi
  printf '    --work-dir "%s" \\\n' "$WORKDIR"
  printf '    --clearml --clearml-project mmdetection \\\n'
  printf '    --cfg-options \n'
  printf '      train_cfg.max_epochs="%s" \\\n' "$EPOCHS"
  printf '      default_hooks.checkpoint.interval=1\n'
  if [[ -n "${EXTRA_CFG_OPTIONS:-}" ]]; then printf '      %s\n' "${EXTRA_CFG_OPTIONS}"; fi
  printf '\n'
} >> "$LOG_FILE"

# --- Post-train: evaluate last/best (visualization hook saves images) ---
# Pick best if exists, else latest/last epoch
CKPT=""
if ls "$WORKDIR"/best_*.pth >/dev/null 2>&1; then
  CKPT=$(ls -t "$WORKDIR"/best_*.pth | head -n1)
elif [[ -f "$WORKDIR/latest.pth" ]]; then
  CKPT="$WORKDIR/latest.pth"
elif ls "$WORKDIR"/epoch_*.pth >/dev/null 2>&1; then
  CKPT=$(ls -t "$WORKDIR"/epoch_*.pth | head -n1)
fi

if [[ -n "$CKPT" ]]; then
  POST_OUT="${RUN_OUTPUT_DIR}/results_posttrain.pkl"
  python tools/test.py "$CONFIG" "$CKPT" \
    --clearml \
    --clearml-project mmdetection \
    --out "$POST_OUT" || true

  # Images saved under work_dir/timestamp/vis_test by DetVisualizationHook
fi
