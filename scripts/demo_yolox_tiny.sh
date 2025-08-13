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
#   bash scripts/demo_yolox_tiny.sh
#   DEVICE=cpu bash scripts/demo_yolox_tiny.sh
#   IMG=path/to.jpg OUTDIR=outputs/demo1 bash scripts/demo_yolox_tiny.sh

export STAGE=inference

IMG=${IMG:-demo/demo.jpg}
OUTDIR_BASE=${OUTDIR_BASE:-${OUT_ROOT}}
RUN_OUTPUT_DIR="${OUTDIR_BASE}/${STAGE}/${DATASET}/${MODEL}/${CONFIG_BASE}/${TAG}/${RUN_ID}"
export RUN_OUTPUT_DIR
OUTDIR=${OUTDIR:-${RUN_OUTPUT_DIR}}
DEMO_CONFIG=${DEMO_CONFIG:-weights/yolox_tiny_8x8_300e_coco.py}
DEMO_WEIGHTS_GLOB=${DEMO_WEIGHTS_GLOB:-weights/yolox_tiny_8x8_300e_coco*.pth}

mkdir -p "$OUTDIR"

python demo/image_demo.py \
  "$IMG" \
  "$DEMO_CONFIG" \
  --weights "$DEMO_WEIGHTS_GLOB" \
  --device "$DEVICE" \
  --out-dir "$OUTDIR"
