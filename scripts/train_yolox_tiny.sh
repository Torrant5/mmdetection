#!/usr/bin/env bash
set -euo pipefail

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

python tools/train.py "$CONFIG" \
  ${AMP_FLAG} \
  --work-dir "$WORKDIR" \
  ${CLEARML_FLAGS} \
  --cfg-options \
    load_from="$WEIGHTS" \
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
  if [[ -n "$CLEARML_FLAGS" ]]; then printf '    %s \\\n' "$CLEARML_FLAGS"; fi
  printf '    --cfg-options \n'
  printf '      load_from="%s" \\\n' "$WEIGHTS"
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
    ${CLEARML_FLAGS} \
    --out "$POST_OUT" || true

  # Images saved under work_dir/timestamp/vis_test by DetVisualizationHook
fi
