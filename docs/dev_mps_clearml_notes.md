# MPS + ClearML 運用メモ（mmcv フォーク方針を含む）

## mmcv フォーク（MPS対応について）
- 背景: Apple Silicon (MPS) 対応のため、上流 `mmcv` に先行して必要な互換修正を含むフォークを利用しています（本リポジトリ直下にベンダリング: `mmcv/`）。
- 目的: CUDA 前提の実装や autocast の挙動などで MPS 実行が阻害される箇所の回避・安定化（暫定措置）。
- 方針: `mmcv/` はスナップショット扱い。原則ここで直接編集せず、フォーク側（外部リポジトリ）で更新→取り込み。
- 更新手順（推奨）:
  1) 自フォークの `mmcv` を上流と同期し、MPS向けパッチ維持ブランチを用意
  2) 本リポジトリと組み合わせて MPS/CPU で最小動作確認（`make demo-yolox-tiny` / `make test-yolox-tiny`）
  3) 問題なければ `mmcv/` ディレクトリを差し替え（必要に応じてコミット）
- 将来: Git サブモジュールやパッケージ配布へ切替し、ベンダリング解消を目標。

## ClearML 連携の要点
- `tools/{train,test}.py` に `--clearml` オプションを追加済み。`--clearml-project` / `--clearml-task` も指定可。
- 自動タグ付与: `stage, dataset, model, tag, config_base, device`（環境変数から収集）。
- TensorBoard を `visualizer` に追加しておくと、スカラーが ClearML 側に取り込まれやすい。

## MPS 実行の要点
- `PYTORCH_ENABLE_MPS_FALLBACK=1` をデフォルトで有効化（未対応演算はCPUへフォールバック）。
- AMP は CUDA 時のみ有効化推奨（MPS/CPU では `USE_AMP=0`）。
- 一部 autocast は `torch.amp.autocast(device_type=...)` を使用（CUDA/MPS/CPUを明示）。

