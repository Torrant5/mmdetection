# CLAUDE.md - Claude Code Agent向けプロジェクト情報

## プロジェクト概要
mmdetectionを使用した物体検出モデルの学習・評価環境

## リポジトリの独立性について
**重要**: このリポジトリはmmdetectionのフォークですが、PRは本フォークリポジトリ内で完結させ、元のリポジトリ（open-mmlab/mmdetection）には影響を与えません。
- すべての変更は本フォーク内でのみ管理
- 独自の機能追加や改修は自由に実施可能
- upstream（元リポジトリ）への貢献は別途検討

## AGENTS.mdとの関係
`.codex/AGENTS.md`は同僚との情報同期用のメモです。CLAUDE.mdとAGENTS.mdは常に同じ視座で開発できるよう、ほぼ同じ内容が記載されるべきものです。
- CLAUDE.md: Claude Code Agent向けの詳細な説明
- AGENTS.md: 同僚との共有用、両者で同じ認識を持つための同期ドキュメント
- 重要な変更は両方のファイルに反映すること

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

## 学習実行ガイドライン

### 設定ファイル作成に関する重要な方針
**新しい設定ファイルを無闇に作成しないでください。** 基本的に以下の方針で対応してください：

1. **既存の設定ファイルを活用**
   - 新しい設定ファイルを作成する前に、既存のものを利用できないか確認
   - `configs/yolox/ytiny_*_train.py`や`configs/yolox/ytiny_*_test.py`などがすでに存在

2. **コマンドラインオプションで対応**
   - パラメータの変更は`EXTRA_CFG_OPTIONS`環境変数を使用
   - 例：`EXTRA_CFG_OPTIONS="train_cfg.val_interval=1 default_hooks.logger.interval=20"`
   - エポック数、バッチサイズ、学習率などはコマンドラインで調整可能

3. **設定ファイル作成が必要な場合**
   - モデル構造の大幅な変更
   - 新しいデータセットへの対応
   - 上記の場合でも、既存ファイルのコピー・編集を優先

### tmuxセッション管理
エージェント内で長時間の学習を実行する際は、以下の方針でtmuxセッションを使用してください：

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

3. **外部からの参照**
   ```bash
   # セッション一覧確認
   tmux ls
   
   # セッションにアタッチ
   tmux attach -t "claude-yolox-ep15-2cls"
   
   # ログ確認
   tmux capture-pane -t "claude-yolox-ep15-2cls" -p
   ```

4. **セッションの削除**
   **重要**: 処理が完了したtmuxセッションは、システムリソースを解放するため必ず削除してください。
   削除前に必ず実行状況を確認し、実行中のタスクがある場合は適切に停止してください。
   
   ```bash
   # 1. セッションの実行状況を確認
   tmux capture-pane -t "claude-<セッション名>" -p | tail -20
   
   # 2. 実行中のプロセスを確認
   tmux send-keys -t "claude-<セッション名>" C-c  # Ctrl-Cで実行中のコマンドを停止（必要な場合）
   sleep 2  # 停止を待つ
   
   # 3. セッションの最終状態を確認
   tmux capture-pane -t "claude-<セッション名>" -p | tail -5
   
   # 4. 問題なければセッションを削除
   tmux kill-session -t "claude-<セッション名>"
   ```
   
   **安全な一括削除の例**:
   ```bash
   # claudeプレフィックスのセッションを確認
   for session in $(tmux ls | grep "^claude-" | cut -d: -f1); do
     echo "Checking session: $session"
     # 最新の出力を確認
     tmux capture-pane -t "$session" -p | tail -5
     echo "---"
   done
   
   # 確認後、個別に削除（自動削除は推奨しません）
   # tmux kill-session -t "claude-<確認済みセッション名>"
   ```
   
   **注意事項**:
   - 学習やテストが実行中の場合は、完了を待つか、必要に応じてCtrl-Cで安全に停止
   - チェックポイントの保存状況を確認してから削除
   - 重要な結果やログが必要な場合は、削除前に保存

### 学習コマンド例

#### 基本的な学習（ClearML有効）
```bash
# 2クラス、15エポック学習（ClearML有効、AMP無効）
EPOCHS=15 USE_AMP=0 \
  EXTRA_CFG_OPTIONS="train_cfg.val_interval=1 default_hooks.logger.interval=20" \
  make train-ytiny-20250709-taikoB-img2

# 80クラス学習（ClearML有効）
EPOCHS=15 make train-ytiny-20250709-taikoB-img80
```

#### ClearMLを無効化したい場合
```bash
# ClearMLを明示的に無効化
EPOCHS=15 CLEARML_FLAGS= make train-ytiny-20250709-taikoB-img2
```

#### テスト実行
```bash
# 事前学習済みモデルでテスト（ClearML有効）
make test-ytiny-20250709-taikoB-img80

# 最新チェックポイントでテスト
make test-ytiny-20250709-taikoB-img2-last
```

## 重要な設定ファイル

### 学習設定
- `configs/yolox/ytiny_20250709_toshitaiko_seat-knockB_img2_train.py` - 2クラス学習用
- `configs/yolox/ytiny_20250709_toshitaiko_seat-knockB_img80_train.py` - 80クラス学習用

### 環境変数
- `EPOCHS`: 学習エポック数
- `CLEARML_FLAGS`: ClearML統合の制御
  - デフォルト: `--clearml --clearml-project mmdetection`（有効）
  - 無効化: `CLEARML_FLAGS=`（空文字列を設定）
- `USE_AMP`: 自動混合精度の有効/無効（0で無効化）
- `EXTRA_CFG_OPTIONS`: 追加の設定オプション

## 出力ディレクトリ構造
```
output/
├── train/
│   └── <DATASET>/<MODEL>/<CONFIG>/<TAG>/<RUN_ID>/
│       ├── epoch_*.pth  # チェックポイント
│       ├── vis_data/     # 可視化結果
│       └── scalars.json  # メトリクス
└── test/
    └── <DATASET>/<MODEL>/<CONFIG>/<TAG>/<RUN_ID>/
        ├── results.pkl   # 評価結果
        └── vis_test/     # テスト画像の可視化
```

## トラブルシューティング

### ModuleNotFoundError: pycocotools
```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mmdet
```

### PyTorch 2.6 weights_only エラー
tools/test.pyとtools/train.pyに必要なnumpy dtypeが追加済み。
新しいdtypeエラーが出た場合は、同様にadd_safe_globalsリストに追加。

### ClearMLエラー
ClearML統合を無効化：
```bash
CLEARML_FLAGS= make train-...
```

## 推奨ワークフロー

1. **データ準備**
   ```bash
   make splits-20250709-taikoB-img2  # データ分割生成
   ```

2. **学習実行（tmux使用、ClearML有効）**
   ```bash
   # セッション作成と学習開始（ClearML有効）
   tmux new-session -d -s "claude-yolox-ep15-2cls-clearml"
   tmux send-keys -t "claude-yolox-ep15-2cls-clearml" \
     "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mmdet" C-m
   tmux send-keys -t "claude-yolox-ep15-2cls-clearml" \
     "EPOCHS=15 USE_AMP=0 EXTRA_CFG_OPTIONS=\"train_cfg.val_interval=1 default_hooks.logger.interval=20\" make train-ytiny-20250709-taikoB-img2" C-m
   ```

3. **進捗確認**
   ```bash
   # 特定のセッション名を指定（ワイルドカードは使用不可）
   tmux capture-pane -t "claude-yolox-ep15-2cls-clearml" -p | grep "Epoch"
   ```

4. **評価実行**
   ```bash
   make test-ytiny-20250709-taikoB-img2-last
   ```

## 2クラス学習の重要な注意事項

### カテゴリIDとクラス名
2クラス（person, sports ball）の学習では、**COCO標準のカテゴリIDとクラス名を保持**しています：
- **person**: ID 1（COCO標準）
- **sports ball**: ID 33（COCO標準、スペース付き）

重要：クラス名は`sports ball`（スペース付き）が正しい形式です。これは`CocoDataset.METAINFO`の定義に合わせています。

この設定により、事前学習済みモデルのsports ballに関する知識を最大限活用できます。

### 修正済みファイル
- `tools/dataset_utils/build_dataset_pipeline.py`: ID remappingを削除し、COCO標準IDを保持、クラス名を`sports ball`に修正
- アノテーションファイル: IDを2から33に、クラス名を`sports_ball`から`sports ball`に修正済み
- 設定ファイル: `ytiny_*_img2_*.py`のクラス名を`sports ball`に修正済み

### 検証スクリプト
```bash
# カテゴリIDが正しいことを確認
python tools/dataset_utils/verify_category_mapping.py
```

## ツールディレクトリ構成

### scripts/ - 実行スクリプト
- `train_yolox_tiny.sh` - YOLOX-Tiny学習用スクリプト
- `test_yolox_tiny.sh` - YOLOX-Tinyテスト用スクリプト
- `demo_yolox_tiny.sh` - デモ実行スクリプト
- `common.sh` - 共通設定（データパス、モデル設定）
- `setup_symlinks.sh` - シンボリックリンク設定

### tools/dataset_utils/ - データセット処理ツール
- `build_dataset_pipeline.py` - データセット分割とアノテーション生成（COCO標準ID保持対応済み）
- `fix_category_ids.py` - **[DEPRECATED]** カテゴリIDを連番化（33→2）※使用非推奨、COCO標準ID(1,33)を保持すべき
- `fix_class_names.py` - クラス名を`sports ball`形式に修正
- `verify_category_mapping.py` - カテゴリマッピングの検証（推奨）

## PRレビュー対応ガイドライン

### 重要: レビューコメントへの対応方針
PRレビューを受けた際は、以下の手順で対応すること：

1. **全コメントの確認**
   - 全体的なPRコメントだけでなく、個別のラインコメントまで確認する
   - **ラインコメントの確認方法**：
     ```bash
     # 全体的なコメントとラインコメントを含めて表示
     gh pr view <PR番号> --comments
     
     # ラインコメントをJSON形式で取得（詳細情報付き）
     gh api repos/<owner>/<repo>/pulls/<PR番号>/comments --jq '.[] | {id: .id, path: .path, line: .line, body: .body}'
     
     # 特定ファイルへのコメントを検索
     gh pr view <PR番号> --comments | grep -A10 "<ファイル名>"
     ```

2. **対応判断基準**
   - **技術的正確性の検証**: レビュワーの指摘が技術的に正しいか確認する
   - **プロジェクト方針との整合性**: プロジェクトの設計方針に合致するか確認する
   - **必要性の評価**: 実際に修正が必要か、現状で問題ないか判断する

3. **対応内容の記録**
   - 修正した項目: 具体的な変更内容を記載
   - 修正しない項目: 技術的根拠を含めて理由を説明

4. **ラインコメントへの個別返信**
   - **重要**: 全体的なPRコメントだけでなく、各ラインコメントに個別に返信する
   - **返信方法**：
     ```bash
     # ラインコメントのIDを取得
     gh api repos/<owner>/<repo>/pulls/<PR番号>/comments --jq '.[] | {id: .id, path: .path, line: .line}'
     
     # 特定のラインコメントに返信
     gh api -X POST repos/<owner>/<repo>/pulls/<PR番号>/comments/<コメントID>/replies \
       -f body="返信内容"
     ```
   - **返信のポイント**：
     - 修正した場合: 「✅ 修正しました（コミット: <SHA>）」で始める
     - 修正しない場合: 技術的根拠を明確に説明
     - コード例を含める場合は適切なMarkdownフォーマットを使用

5. **特にmmdetection固有の注意点**
   - **カテゴリID**: mmdetectionのCocoDatasetは連番である必要はなく、COCO標準ID（例: person=1, sports ball=33）をそのまま使用可能
   - **データ拡張**: ファインチューニング時のデータ拡張強度は、データセットサイズと学習目的により調整
   - **設定の継承**: ベース設定からの継承とオーバーライドの仕組みを理解した上で判断

### 大量のレビューコメントへの対応方法

コメントが多い場合は、一旦ファイルに書き出してから対応することを推奨：

1. **コメントの全取得と整理**
   ```bash
   # .notes ディレクトリを作成（未作成の場合）
   mkdir -p .notes
   
   # 全コメントをJSONで取得して保存
   gh api repos/<owner>/<repo>/pulls/<PR番号>/comments --paginate --jq '.[] | {
     id: .id,
     path: .path,
     line: .line,
     body: (.body | split("\n")[0:3] | join(" ")),
     created_at: .created_at,
     user: .user.login,
     in_reply_to: .in_reply_to_id
   }' | jq -s 'sort_by(.created_at)' > .notes/pr_comments.json
   ```

2. **未対応コメントのMarkdown化**
   ```bash
   # .notes/pending_pr_comments.md に未対応リストを作成
   cat > .notes/pending_pr_comments.md << 'EOF'
   # PR #<番号> 未対応コメント一覧
   
   ## 1. <ファイル名> - <概要>
   - **ID**: <コメントID>
   - **ファイル**: <ファイルパス>
   - **行**: <行番号>
   - **内容**: <コメント概要>
   - **詳細**: <対応が必要な内容>
   
   ## 対応状況
   - [ ] 1. <項目1>
   - [ ] 2. <項目2>
   EOF
   ```

3. **TodoWriteツールとの連携**
   - 各コメント対応をTodoWriteツールで管理
   - 対応完了したらチェックマークを付けて追跡

4. **対応後のクリーンアップ**
   ```bash
   # 対応完了後は.notesディレクトリの作業ファイルを削除
   rm -f .notes/pr_comments.json .notes/pending_pr_comments.md
   ```

この方法により：
- 大量のコメントを見落とさない
- 対応状況を明確に管理できる
- チーム内で進捗を共有しやすい

## PRレビュー対応の実例

### ラインコメント対応例（PR #2より）
```bash
# 1. すべてのラインコメントを確認
gh api repos/Torrant5/mmdetection/pulls/2/comments --jq '.[] | {id: .id, path: .path, line: .line}'

# 2. 修正が必要なコメントに返信（例: fix_category_ids.py）
gh api -X POST repos/Torrant5/mmdetection/pulls/2/comments/2272721219/replies \
  -f body="✅ レビュワーの指摘が正しいです。修正しました（コミット: c8d07f67）

mmdetectionのCocoDatasetは内部でcat2labelマッピングを使用し、COCO標準IDを自動的に連番インデックスに変換します。

対応：
1. スクリプトに[DEPRECATED]マーク追加
2. COCO標準ID維持の重要性を明記"

# 3. 修正不要なコメントに返信（例: MultiImageMixDataset）
gh api -X POST repos/Torrant5/mmdetection/pulls/2/comments/2272721165/replies \
  -f body="現状維持で問題ありません。

MultiImageMixDatasetはYOLOXフレームワークの必須要件です。
YOLOXModeSwitchHookと組み合わせることで、ファインチューニング時にMosaic/MixUpを適切に無効化できます。"
```

## メモ
- 学習時は必ずconda環境をアクティベートすること
- 長時間の学習はtmuxセッションで実行し、外部から監視可能にすること
- チェックポイントは3エポックごとに保存される（max_keep_ckpts=3）
- 2クラス学習では必ずCOCO標準のカテゴリIDを使用すること
- **PRレビュー対応は、ラインコメントまで含めて個別に返信してから完了とすること**