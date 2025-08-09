#!/usr/bin/env bash
set -euo pipefail

# Download RTMDet-tiny COCO weights to a local ignored folder.
# Intended for local testing; files in ./weights are git-ignored.

URL="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
OUT_DIR="weights"
OUT_FILE="${OUT_DIR}/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"

mkdir -p "${OUT_DIR}"
if [ -f "${OUT_FILE}" ]; then
  echo "Weights already exist: ${OUT_FILE}"
else
  echo "Downloading weights to ${OUT_FILE} ..."
  curl -L "${URL}" -o "${OUT_FILE}"
  echo "Saved: ${OUT_FILE}"
fi

