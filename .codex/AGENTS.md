# AGENTS.md - 同僚との同期用プロジェクト情報

## CLAUDE.mdとの関係
このファイルはCLAUDE.mdと同期して管理されます。CLAUDE.mdはClaude Code Agent向けの説明で、本ファイルは同僚との情報共有用です。両者で同じ視座を持って開発できるよう、ほぼ同じ内容を記載します。
- AGENTS.md: 同僚との共有・同期用ドキュメント
- CLAUDE.md: Claude Code Agent向けの詳細説明
- 重要な変更は両方のファイルに反映すること

## プロジェクト概要
mmdetectionを使用した物体検出モデルの学習・評価環境

## リポジトリの独立性について
**重要**: このリポジトリはmmdetectionのフォークですが、PRは本フォークリポジトリ内で完結させ、元のリポジトリ（open-mmlab/mmdetection）には影響を与えません。
- すべての変更は本フォーク内でのみ管理
- 独自の機能追加や改修は自由に実施可能
- upstream（元リポジトリ）への貢献は別途検討

## 環境設定
- Python環境: conda環境 `mmdet` を使用
- 活性化コマンド: `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mmdet`

## PyTorch 2.6対応
PyTorch 2.6では`weights_only`のデフォルト値が`True`に変更されました。以下のnumpyタイプを安全グローバルリストに追加済み：
- `numpy.dtypes.Float64DType`
- `numpy.dtypes.Int64DType`

対応ファイル：
- `tools/test.py`
- `tools/train.py`

## 作業ブランチ
- 本件の統合作業ブランチ: `feat/mps-clearml-env`

## 学習実行ガイドライン

### tmuxセッション管理
長時間の学習を実行する際は、tmuxセッションを使用：

1. **セッション命名規則**
   ```bash
   tmux new-session -d -s "claude-<タスク名>-<条件>"
   # 例: tmux new-session -d -s "claude-yolox-ep15-2cls"
   ```

2. **学習実行例**
   ```bash
   # tmuxセッション内で学習を開始
   tmux send-keys -t "claude-yolox-ep15-2cls" \
     "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mmdet" C-m
   tmux send-keys -t "claude-yolox-ep15-2cls" \
     "EPOCHS=15 make train-ytiny-20250709-taikoB-img2" C-m
   ```

3. **セッションの削除（重要）**
   処理完了後は必ず削除：
   ```bash
   # 実行状況を確認してから削除
   tmux capture-pane -t "claude-<セッション名>" -p | tail -20
   tmux kill-session -t "claude-<セッション名>"
   ```

## 重要な設定ファイル

### 学習設定
- `configs/yolox/ytiny_20250709_toshitaiko_seat-knockB_img2_finetune.py` - 2クラスfinetuning用
- `configs/yolox/ytiny_20250709_toshitaiko_seat-knockB_img2_train.py` - 2クラス学習用
- `configs/yolox/ytiny_20250709_toshitaiko_seat-knockB_img80_train.py` - 80クラス学習用

### 環境変数
- `EPOCHS`: 学習エポック数
- `CLEARML_FLAGS`: ClearML統合の制御（デフォルト有効、`CLEARML_FLAGS=`で無効化）
- `USE_AMP`: 自動混合精度の有効/無効（0で無効化）
- `EXTRA_CFG_OPTIONS`: 追加の設定オプション

## 出力ディレクトリ構造
```
output/  # work_dirsではなくoutput/に統一
├── train/
│   └── <DATASET>/<MODEL>/<CONFIG>/<TAG>/<RUN_ID>/
│       ├── epoch_*.pth  # チェックポイント
│       ├── vis_data/     # 可視化結果
│       └── scalars.json  # メトリクス
└── test/
    └── <DATASET>/<MODEL>/<CONFIG>/<TAG>/<RUN_ID>/
        ├── results.pkl   # 評価結果
        └── vis_test/     # テスト画像の可視化（自動出力）
```

## ツール構成

### tools/dataset_utils/ - データセット処理専用ツール
- `build_dataset_pipeline.py`: データセット分割とアノテーション生成
  - **重要**: COCO標準ID保持対応済み（person: ID 1, sports ball: ID 33）
  - Make ターゲット: `make splits-20250709-taikoB-img80` / `make splits-20250709-taikoB-img2`
- `fix_category_ids.py`: カテゴリIDをCOCO標準に修正（ID 2 → 33）
- `fix_class_names.py`: クラス名を`sports ball`形式に修正
- `verify_category_mapping.py`: カテゴリマッピングの検証
- `tools/misc/transfer_coco_weights_to_2class.py`: 80クラス→2クラス重み転送

### scripts/ - 実行スクリプト
- `train_yolox_tiny.sh`: 学習実行（ClearML常時有効、load_from上書き削除）
- `test_yolox_tiny.sh`: 評価実行（--work-dir追加でwork_dirs作成防止）
- `common.sh`: 共通設定（データパス、モデル設定）
- `demo_yolox_tiny.sh`: デモ実行
- `setup_symlinks.sh`: シンボリックリンク設定

## 実行例（Makefile経由）

### 基本的な学習（ClearML有効）
```bash
# 2クラス、15エポック学習（finetuning設定使用）
EPOCHS=15 USE_AMP=0 \
  EXTRA_CFG_OPTIONS="train_cfg.val_interval=1 default_hooks.logger.interval=20" \
  make train-ytiny-20250709-taikoB-img2

# 80クラス学習
EPOCHS=15 make train-ytiny-20250709-taikoB-img80
```

### ClearMLを無効化したい場合
```bash
# ClearMLを明示的に無効化
EPOCHS=15 CLEARML_FLAGS= make train-ytiny-20250709-taikoB-img2
```

### テスト実行
```bash
# 事前学習済みモデルでテスト（可視化画像自動出力）
make test-ytiny-20250709-taikoB-img80

# 最新チェックポイントでテスト
make test-ytiny-20250709-taikoB-img2-last
```

## 2クラス学習の重要事項

### カテゴリIDとクラス名
2クラス（person, sports ball）では**COCO標準のカテゴリIDとクラス名を保持**：
- **person**: ID 1（COCO標準）
- **sports ball**: ID 33（COCO標準、スペース付き）

クラス名は`sports ball`（スペース付き）が正しい形式。これにより事前学習済みモデルの知識を最大限活用。

### Finetuning設定の特徴
- 学習率: 1e-5（通常の1/1000、保守的）
- MixUp/Mosaic: 無効化（num_last_epochs=30）
- 解像度: 3040x3040（小物体検出対応）
- 転送済み重み: `weights/yolox_tiny_2class_transferred_new.pth`使用

## 入口リンク（詳細ドキュメント）
- 実行メモ（MPS + ClearML 概要）: `docs/runs_yolox_tiny_clearml_mps.md`
- カスタムデータでの実行例: `docs/runs_yolox_tiny_custom_images.md`
- E2E 手順（分割→評価→学習→評価）: `docs/howto_e2e_mps_clearml.md`
- MPS/ClearML の運用ノート（mmcv フォーク方針）: `docs/dev_mps_clearml_notes.md`

## メモ
- 学習時は必ずconda環境をアクティベートすること
- 長時間の学習はtmuxセッションで実行し、外部から監視可能にすること
- work_dirsではなくoutput/ディレクトリを使用（scripts/で設定済み）
- テスト時は自動的に可視化画像が出力される（visualization hook追加済み）
- 2クラス学習では必ずCOCO標準のカテゴリIDを使用すること