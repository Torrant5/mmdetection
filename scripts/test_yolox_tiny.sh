#!/usr/bin/env bash
set -euo pipefail

DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$DIR/common.sh"

# 使用例:
#   bash scripts/test_yolox_tiny.sh
#   OUT=outputs/yolox_tiny_val.pkl bash scripts/test_yolox_tiny.sh

export STAGE=test

# 出力ディレクトリ構築
RUN_OUTPUT_DIR="${OUT_ROOT}/${STAGE}/${DATASET}/${MODEL}/${CONFIG_BASE}/${TAG}/${RUN_ID}"
export RUN_OUTPUT_DIR
mkdir -p "$RUN_OUTPUT_DIR"

OUT=${OUT:-${RUN_OUTPUT_DIR}/results.pkl}

python tools/test.py "$CONFIG" "$WEIGHTS" \
  ${CLEARML_FLAGS} \
  --out "$OUT"

# --- Run log ---
LOG_FILE=".notes/py/RUN_LOG.md"
mkdir -p .notes/py
{
  printf '## %s test\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf -- '- CONFIG: %s\n' "$CONFIG"
  printf -- '- WEIGHTS: %s\n' "$WEIGHTS"
  printf -- '- DEVICE: %s\n' "${DEVICE:-}"
  printf -- '- DATASET: %s  MODEL: %s  TAG: %s\n' "${DATASET:-}" "${MODEL:-}" "${TAG:-}"
  printf -- '- OUT: %s\n' "$OUT"
  printf -- '- RUN_OUTPUT_DIR: %s\n' "${RUN_OUTPUT_DIR:-}"
  printf -- '- Command:\n'
  printf '  python tools/test.py "%s" "%s" \\\n' "$CONFIG" "$WEIGHTS"
  if [[ -n "$CLEARML_FLAGS" ]]; then printf '    %s \\\n' "$CLEARML_FLAGS"; fi
  printf '    --out "%s"\n' "$OUT"
  printf '\n'
} >> "$LOG_FILE"

# Prediction images are saved by DetVisualizationHook configured in the config
