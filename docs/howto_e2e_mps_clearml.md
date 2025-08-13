# E2E 手順: MPS + ClearML で YOLOX-Tiny を学習/評価

目的: カスタム分割の生成 → 80クラス評価 → 2クラス学習→評価 → 成果物/可視化/メトリクス確認までを一気通貫で確認します。

前提
- COCO 評価用: `pycocotools` が導入済み
- 必要なら外部USBへのリンク作成（出力/データを外付けに集約）
- ClearML を使う場合は `clearml-init` 済み（使わない場合は `CLEARML_FLAGS=` で無効化）

1) 外部リンク（任意）
```
make setup-links
# 別ディスク例
USB_ROOT="/Volumes/otherdisk" make setup-links
```

2) 分割の生成（images=80クラス／images2=2クラス）
```
# 80クラス（images）
python tools/misc/build_dataset_pipeline.py \
  data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/instances_default.json \
  data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200 \
  --variant images \
  --num-images 200 \
  --train-ratio 0.8 \
  --copy-mode symlink

# 2クラス（images2: person, sports_ball）
python tools/misc/build_dataset_pipeline.py \
  data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/instances_default.json \
  data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200 \
  --variant images --images-2class \
  --num-images 200 \
  --train-ratio 0.8 \
  --copy-mode symlink
```

3) 80クラスの評価（学習済み重みで動作確認）
```
# ClearMLを使わない例（使う場合は CLEARML_FLAGS を外す）
CLEARML_FLAGS= \
  make test-ytiny-20250709-taikoB-img80
```

4) 2クラスの学習→評価
```
# 例: 15エポック学習、毎epoch評価、ログ出力を少し詳細化
EPOCHS=15 CLEARML_FLAGS= USE_AMP=0 \
  EXTRA_CFG_OPTIONS="train_cfg.val_interval=1 default_hooks.logger.interval=20" \
  make train-ytiny-20250709-taikoB-img2

# 学習後、最新の重みで評価（自動検出）
CLEARML_FLAGS= \
  make test-ytiny-20250709-taikoB-img2-last
```

5) 期待される成果物と確認ポイント
- 出力配置: `output/{inference|test|train}/{DATASET}/{MODEL}/{CONFIG_BASE}/{TAG}/{RUN_ID}/`
- 可視化画像: テスト時は `work_dirs/<config>/<timestamp>/vis_test/`（DetVisualizationHook）
- メトリクス: `tools/test.py --out` の pkl、ClearML を有効化した場合は Task 上の Scalars/Plots/Images
- 実行ログ: `.notes/py/RUN_LOG.md` にコマンドとメタ情報が追記されます

6) MPS/AMP について
- 既定で `PYTORCH_ENABLE_MPS_FALLBACK=1` を有効化し、未対応演算は自動でCPUフォールバック
- AMP は CUDA 時のみ有効化（MPS/CPU では `USE_AMP=0` 推奨）

注: 複数行コマンドは「行末にのみバックスラッシュ」を置き、次行で継続してください。1行で実行する場合は環境変数を前置して書けます（例: `CLEARML_FLAGS= make test-...`）。

