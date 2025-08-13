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
#   bash scripts/test_yolox_tiny.sh
#   OUT=outputs/yolox_tiny_val.pkl bash scripts/test_yolox_tiny.sh

export STAGE=test

# 出力ディレクトリ構築
RUN_OUTPUT_DIR="${OUT_ROOT}/${STAGE}/${DATASET}/${MODEL}/${CONFIG_BASE}/${TAG}/${RUN_ID}"
export RUN_OUTPUT_DIR
mkdir -p "$RUN_OUTPUT_DIR"

OUT=${OUT:-${RUN_OUTPUT_DIR}/results.pkl}

python tools/test.py "$CONFIG" "$WEIGHTS" \
  --clearml \
  --clearml-project mmdetection \
  --work-dir "$RUN_OUTPUT_DIR" \
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
  printf '    --clearml --clearml-project mmdetection \\\n'
  printf '    --out "%s"\n' "$OUT"
  printf '\n'
} >> "$LOG_FILE"

# Prediction images are saved by DetVisualizationHook configured in the config
