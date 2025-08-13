# AGENTS.md - プロジェクト同期用ドキュメント

## このドキュメントについて
このファイルはCLAUDE.mdと同期して管理されます。開発チーム内で同じ視座を持って作業できるよう、重要な情報を共有します。
- AGENTS.md: チーム内共有・同期用（このファイル）
- CLAUDE.md: Claude Code Agentが参照する詳細説明
- 重要な変更は両方のファイルに反映すること

## プロジェクト概要
mmdetectionを使用した物体検出モデルの学習・評価環境

## リポジトリ運用方針
**重要**: このリポジトリはmmdetectionのフォークですが、PRは本フォークリポジトリ内で完結させます。
- すべての変更は本フォーク内でのみ管理
- 独自の機能追加や改修は自由に実施
- upstream（open-mmlab/mmdetection）への貢献は別途検討

## 環境設定
- Python環境: conda環境 `mmdet` を使用
- 活性化コマンド: `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mmdet`

## PyTorch 2.6対応
PyTorch 2.6では`weights_only`のデフォルト値が`True`に変更されたため、以下の対応を実施済み：
- `numpy.dtypes.Float64DType`と`Int64DType`を安全グローバルリストに追加
- 対応ファイル: `tools/test.py`, `tools/train.py`

## 作業ブランチ
- 統合作業ブランチ: `feat/mps-clearml-env`

## 学習実行の運用

### tmuxセッション利用
長時間学習では以下の命名規則でtmuxセッションを作成：

```bash
# セッション作成
tmux new-session -d -s "claude-<タスク名>-<条件>"

# 学習開始
tmux send-keys -t "claude-yolox-ep15-2cls" \
  "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mmdet" C-m
tmux send-keys -t "claude-yolox-ep15-2cls" \
  "EPOCHS=15 make train-ytiny-20250709-taikoB-img2" C-m

# 完了後は必ず削除
tmux capture-pane -t "claude-<セッション名>" -p | tail -20
tmux kill-session -t "claude-<セッション名>"
```

## 重要な設定ファイル

### 学習設定
- `configs/yolox/ytiny_*_img2_finetune.py` - 2クラスfinetuning用（推奨）
- `configs/yolox/ytiny_*_img2_train.py` - 2クラス通常学習用
- `configs/yolox/ytiny_*_img80_train.py` - 80クラス学習用

### 主要な環境変数
- `EPOCHS`: エポック数
- `CLEARML_FLAGS`: ClearML制御（デフォルト有効、空文字列で無効化）
- `USE_AMP`: 自動混合精度（0で無効化）
- `EXTRA_CFG_OPTIONS`: 追加設定オプション

## ディレクトリ構造（重要変更）
**注意**: `work_dirs/`ではなく`output/`に統一しました。

```
output/
├── train/
│   └── <DATASET>/<MODEL>/<CONFIG>/<TAG>/<RUN_ID>/
│       ├── epoch_*.pth
│       ├── vis_data/
│       └── scalars.json
└── test/
    └── <DATASET>/<MODEL>/<CONFIG>/<TAG>/<RUN_ID>/
        ├── results.pkl
        └── vis_test/  # 自動出力される可視化画像
```

## ツール構成

### データセット処理（tools/dataset_utils/）
- `build_dataset_pipeline.py`: データセット分割（COCO標準ID保持対応済み）
- `fix_category_ids.py`: ID修正（2→33）
- `fix_class_names.py`: クラス名修正（sports_ball→sports ball）
- `verify_category_mapping.py`: マッピング検証

### 重み転送（tools/misc/）
- `transfer_coco_weights_to_2class.py`: 80クラス→2クラス重み転送

### 実行スクリプト（scripts/）
- `train_yolox_tiny.sh`: 学習実行（ClearML常時有効化済み）
- `test_yolox_tiny.sh`: 評価実行（work_dirs作成防止済み）
- `common.sh`: 共通設定

## よく使うコマンド

### 学習
```bash
# 2クラスfinetuning（推奨）
EPOCHS=15 USE_AMP=0 \
  EXTRA_CFG_OPTIONS="train_cfg.val_interval=1 default_hooks.logger.interval=20" \
  make train-ytiny-20250709-taikoB-img2

# ClearML無効化が必要な場合
EPOCHS=15 CLEARML_FLAGS= make train-ytiny-20250709-taikoB-img2
```

### テスト
```bash
# 転送済み重みでテスト
make test-ytiny-20250709-taikoB-img2

# 最新チェックポイントでテスト
make test-ytiny-20250709-taikoB-img2-last
```

## 2クラス学習の注意点

### カテゴリ設定
COCO標準を維持することで事前学習の知識を活用：
- person: ID 1
- sports ball: ID 33（スペース付きが正式）

### Finetuning推奨設定
- 学習率: 1e-5（保守的）
- MixUp/Mosaic: 無効
- 解像度: 3040×3040
- 初期重み: `weights/yolox_tiny_2class_transferred_new.pth`

## 詳細ドキュメント
- MPS + ClearML概要: `docs/runs_yolox_tiny_clearml_mps.md`
- カスタムデータ実行例: `docs/runs_yolox_tiny_custom_images.md`
- E2E手順: `docs/howto_e2e_mps_clearml.md`
- 運用ノート: `docs/dev_mps_clearml_notes.md`

## PRレビュー対応

### レビューコメント対応の基本方針
PRレビューでは以下を徹底：

1. **完全なレビュー確認**
   - 全体コメントとラインコメント両方を確認
   - **確認コマンド例**：
     ```bash
     # 全コメント表示
     gh pr view <PR番号> --comments
     
     # ラインコメントの詳細取得
     gh api repos/<owner>/<repo>/pulls/<PR番号>/comments \
       --jq '.[] | {id: .id, path: .path, line: .line, body: (.body | split("\n")[0])}'
     ```

2. **技術的判断**
   - レビュワーの指摘の技術的妥当性を検証
   - mmdetection固有の仕様を考慮（例: CocoDatasetはCOCO標準IDをそのまま使用可能）
   - 修正要否を根拠と共に判断

3. **対応記録**
   - 修正内容と非対応理由を明確に記録
   - 技術的根拠を含めて返信

4. **ラインコメントへの個別返信**
   - 全体コメントでまとめず、各ラインコメントに個別返信
   - **返信コマンド**：
     ```bash
     # コメントIDを指定して返信
     gh api -X POST repos/<owner>/<repo>/pulls/<PR番号>/comments/<ID>/replies \
       -f body="返信内容"
     ```
   - 修正時は「✅ 修正しました」、維持時は根拠を明記

## 運用上の注意
- conda環境の活性化を忘れずに
- 長時間学習はtmuxで実行し、完了後は削除
- 出力は`output/`に統一（work_dirs不使用）
- テスト時の可視化画像は自動出力される
- COCO標準カテゴリIDを維持すること
- PRレビューはラインコメントまで含めて完全対応すること
