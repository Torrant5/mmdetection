# このフォークの目的と運用メモ（YOLOX-Tiny + 自前データ + ClearML）

このフォークは「MPS + ClearML で YOLOX-Tiny をすばやく試せる最小構成」を提供します。詳細手順やノウハウは docs に集約し、ここでは入口のみ示します。

## 入口リンク
- 実行メモ（MPS + ClearML 概要）: `docs/runs_yolox_tiny_clearml_mps.md`
- カスタムデータでの実行例: `docs/runs_yolox_tiny_custom_images.md`
- E2E 手順（分割→評価→学習→評価）: `docs/howto_e2e_mps_clearml.md`
- MPS/ClearML の運用ノート（mmcv フォーク方針）: `docs/dev_mps_clearml_notes.md`

## 作業ブランチ
- 本件の統合作業ブランチ: `feat/mps-clearml-env`

## ツール構成
### tools/dataset_utils/ - データセット処理専用ツール
- `build_dataset_pipeline.py`: データセット分割とアノテーション生成
  - **重要**: COCO標準ID保持対応済み（person: ID 1, sports ball: ID 33）
  - Make ターゲット: `make splits-20250709-taikoB-img80` / `make splits-20250709-taikoB-img2`
- `fix_category_ids.py`: カテゴリIDをCOCO標準に修正（ID 2 → 33）
- `fix_class_names.py`: クラス名を`sports ball`形式に修正
- `verify_category_mapping.py`: カテゴリマッピングの検証

### scripts/ - 実行スクリプト
- `train_yolox_tiny.sh`, `test_yolox_tiny.sh`: 学習・評価の実行
- `common.sh`: 共通設定（データパス、モデル設定）

## 実行の基本（抜粋）
- 例: 80クラス評価（ClearML無効）
  `CLEARML_FLAGS= make test-ytiny-20250709-taikoB-img80`
- 例: 2クラス学習（15ep, 毎epoch評価, ログ間隔20）
  `EPOCHS=15 USE_AMP=0 EXTRA_CFG_OPTIONS="train_cfg.val_interval=1 default_hooks.logger.interval=20" make train-ytiny-20250709-taikoB-img2`

注: 複数行コマンドは「行末にのみバックスラッシュ」を置き、次行で継続してください。1行で実行する場合は環境変数の前置でも可。
