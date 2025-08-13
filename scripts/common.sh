#!/usr/bin/env bash
set -euo pipefail

# 共通設定（環境変数で上書き可）
CONFIG=${CONFIG:-configs/yolox/yolox_tiny_8xb8-300e_coco_clearml.py}
WORKDIR=${WORKDIR:-work_dirs/yolox_tiny_clearml}
CLEARML_PROJECT=${CLEARML_PROJECT:-mmdetection}

# ClearML 有効化（無効にしたい場合は CLEARML_FLAGS="" を設定）
CLEARML_FLAGS=${CLEARML_FLAGS:---clearml --clearml-project "${CLEARML_PROJECT}"}

# 追加の --cfg-options（例: EXTRA_CFG_OPTIONS="log_level=DEBUG"）
EXTRA_CFG_OPTIONS=${EXTRA_CFG_OPTIONS:-}

# 既定の重みパス（2クラス転移済み）
WEIGHTS=${WEIGHTS:-weights/yolox_tiny_2class_transferred_new.pth}

# MPS/描画などの実行時環境（必要に応じて上書き可）
export PYTHONPATH=${PYTHONPATH:-}
export PYTHONPATH="weights:${PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}

# デバイス指定（demo用。train/testではタグ用途に利用）
export DEVICE=${DEVICE:-mps}

# 出力ルートとメタ情報（環境変数で上書き可）
export OUT_ROOT=${OUT_ROOT:-output}
export DATASET=${DATASET:-coco2017}
export MODEL=${MODEL:-yolox_tiny}

# コンフィグ名（拡張子なし）
CONFIG_BASE=${CONFIG_BASE:-$(basename "${CONFIG}" .py)}
export CONFIG_BASE

# バッチサイズやエポックでタグを構築（指定がなければ省略）
export BATCH=${BATCH:-}
export EPOCHS=${EPOCHS:-}
_tag_dev=${DEVICE}
_tag_bs="${BATCH:+-bs${BATCH}}"
_tag_ep="${EPOCHS:+-ep${EPOCHS}}"
export TAG=${TAG:-${_tag_dev}${_tag_bs}${_tag_ep}}

# 実行ID（タイムスタンプ + git短SHA）
_ts=$(date +%Y-%m-%d_%H%M%S)
_sha=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
export RUN_ID=${RUN_ID:-${_ts}_${_sha}}

# STAGEごとに各スクリプトで設定（train|test|inference）
export STAGE=${STAGE:-}

# 出力ベースディレクトリ（STAGEセット後に各スクリプトで確定）
# export RUN_OUTPUT_DIR は各スクリプト内で確定・エクスポート
