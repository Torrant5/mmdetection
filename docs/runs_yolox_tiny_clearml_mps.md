# 実行メモ: YOLOX-Tiny + ClearML + MPS

MPS（Apple Silicon）とClearMLを使った最小試行のまとめです。環境変数とショートカットを用意しています。

## 前提
- ClearML ローカル: Web=`http://localhost:8080`, API=`http://localhost:8008`, Files=`http://localhost:8081`
- `clearml-init` 済み、`pip install clearml tensorboard`
- COCO2017: `make data-coco` でDL/展開可能（容量・時間に注意）

## 共通の環境変数（scripts/common.sh）
- `PYTHONPATH=weights:$PYTHONPATH`
- `PYTORCH_ENABLE_MPS_FALLBACK=1`
- `MPLBACKEND=Agg`
- `DEVICE=mps`（demoで使用、train/testはタグ用途）
- `CLEARML_FLAGS="--clearml --clearml-project mmdetection"`

## 出力ディレクトリ設計
- ルート: `output/`
- 用途別: `output/{inference|test|train}/`
- メタ: `/{DATASET}/{MODEL}/{CONFIG_BASE}/{TAG}/{RUN_ID}/`
  - 例: `output/train/coco2017/yolox_tiny/yolox_tiny_8xb8-300e_coco_clearml/mps-bs8-ep1/2025-08-09_231245_ab12cd/`
  - `DATASET`, `MODEL`, `TAG` は環境変数で指定
  - `CONFIG_BASE` は `CONFIG` のファイル名（拡張子抜き）
  - `TAG` は `DEVICE` と `BATCH`, `EPOCHS` から自動生成（任意で上書き可）
  - `RUN_ID` はタイムスタンプ+git短SHA

必要に応じて各コマンド実行前に上書き可能です。

## まずはデータ
```
make data-coco
```

### 外部USBのデータ/出力を参照（シンボリックリンク作成）
- 既定マウント: `/Volumes/kohssd`
- USB側の想定構成: `data/`(元データ), `dataset/`(学習用データセット), `result/`(出力格納先)

```
# 既定 (/Volumes/kohssd) を使う
make setup-links

# 変更例（USB_ROOT を指定）
USB_ROOT="/Volumes/otherdisk" make setup-links

# 既存に衝突がある場合は FORCE=1 で置き換え（.bak退避）
FORCE=1 make setup-links
```

- 作られるリンク
  - `output -> ${USB_ROOT}/result`（出力ルートをUSBに集約）
  - `data/<name> -> ${USB_ROOT}/dataset/<name>`（例: `data/coco` など）
  - `data/raw -> ${USB_ROOT}/data`（生データの参照用）

## デモ推論（1枚の画像, MPS）
```
make demo-yolox-tiny
# 例: 別画像/出力に変更
IMG=path/to.jpg TAG=mps-quick MODEL=yolox_tiny DATASET=coco2017 make demo-yolox-tiny
# CPUに切替
DEVICE=cpu make demo-yolox-tiny
```

## 評価（val2017, ClearMLに記録）
```
make test-yolox-tiny
# 例: 出力ファイル名を変更
TAG=mps-valrun MODEL=yolox_tiny DATASET=coco2017 make test-yolox-tiny
```

## 学習（1 epoch動作確認, ClearMLに記録）
```
make train-yolox-tiny
# 例: 10 epochで学習、ログ間隔変更
EPOCHS=10 BATCH=8 TAG=mps-bs8-ep10 EXTRA_CFG_OPTIONS="default_hooks.logger.interval=20" make train-yolox-tiny
```

## 使用コンフィグ
- `configs/yolox/yolox_tiny_8xb8-300e_coco_clearml.py`
  - `TensorboardVisBackend` を有効化（スカラーをClearMLに取り込み）

## データ分割の作成（正式スクリプト）
カスタムデータの分割生成には `tools/misc/build_dataset_pipeline.py` を使用します（旧 `.notes/py/build_dataset_pipeline.py` から移設）。
```
python tools/misc/build_dataset_pipeline.py \
  data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/instances_default.json \
  data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200 \
  --variant images --num-images 200 --train-ratio 0.8 --copy-mode symlink
```

## よくあるポイント
- MPSでの未対応演算は `PYTORCH_ENABLE_MPS_FALLBACK=1` で自動的にCPUへフォールバック
- train/testはデバイス引数を持たないため基本は自動判定（CUDA→CPU）。MPSはPyTorchサポート状況に依存
- ClearMLにスカラーが出ない場合は、TensorBoard導入・上記コンフィグ使用を確認

## ClearMLのフィルタリングに活用
- スクリプトが以下のタグを自動でClearML Taskに付与します（`--clearml`使用時）:
  - `stage:<train|test|inference>`
  - `dataset:<DATASET>`
  - `model:<MODEL>`
  - `tag:<TAG>`
  - `config_base:<CONFIG_BASE>`
  - `device:<DEVICE>`
- タスク名の例: `train:yolox_tiny_8xb8-300e_coco_clearml [coco2017/yolox_tiny/mps-bs8-ep1]`
