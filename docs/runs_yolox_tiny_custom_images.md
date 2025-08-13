# 実行メモ: YOLOX-Tiny（自前 split: images）

自作パイプラインで作成した images 版 split（`splits/images/{train,test}`）を用いて、
1) 学習済みYOLOX-Tinyでテスト（評価）、2) YOLOX-Tinyの学習→テスト を行う手順です。

## 前提
 - `tools/misc/build_dataset_pipeline.py` で images 版 split を作成済み
  - 例: `data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/splits/images/{train,test}`
  - 画像ファイルは train/test に symlink（またはコピー/移動）済み
  - `instances_train.json` / `instances_test.json` は file_name がベース名
- 学習済み重み: `weights/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth`

## 使うConfig
- 短い命名（データセット/モダリティ/クラス数を含む）
  - 80クラス（images）: `configs/yolox/ytiny_20250709_toshitaiko_seat-knockB_img80_{train|test}.py`
  - 2クラス（images2）: `configs/yolox/ytiny_20250709_toshitaiko_seat-knockB_img2_{train|test}.py`
    - images2 は images の2クラス圧縮版（person/sports_ball）

従来のカスタム名（custom_images_*）も残していますが、上記の短い命名を推奨します。

分割作成スクリプトの正式配置は `tools/misc/build_dataset_pipeline.py` です（以前の `.notes/py/build_dataset_pipeline.py` から移設）。

## まずは学習済みモデルで評価（test）
```
make test-ytiny-20250709-taikoB-img80
# 任意のタグや出力先指定
TAG=mps-eval DATASET=20250709_toshitaiko_seat-knockB_img80 make test-ytiny-20250709-taikoB-img80
```

内部的に実行されるのは以下相当です（環境変数は scripts/common.sh を参照）。
```
CONFIG=configs/yolox/yolox_tiny_custom_images_test.py \
WEIGHTS=weights/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth \
python tools/test.py $CONFIG $WEIGHTS --out output.pkl
```

## 次に学習→評価
```
# 例: 5 epoch 学習（80クラス／images）
EPOCHS=5 make train-ytiny-20250709-taikoB-img80

# 終了後に同じ重みで評価（WEIGHTSを学習出力に切替して実行）
# 例: work_dir は output/train/.../ に作成されます
# WEIGHTS 絶対/相対パスを指定
WEIGHTS=output/train/.../epoch_5.pth make test-ytiny-20250709-taikoB-img80
```

学習コマンドの内部的なイメージ:
```
CONFIG=configs/yolox/ytiny_20250709_toshitaiko_seat-knockB_img80_train.py \
WEIGHTS=weights/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth \
EPOCHS=5 \
python tools/train.py $CONFIG --amp --work-dir work_dirs/run \
  --cfg-options default_hooks.checkpoint.interval=1
```

### 2クラス（images2）で学習・評価
```
# 2クラス（person, sports_ball）に圧縮した images2 を使用
python tools/misc/build_dataset_pipeline.py \
  data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/instances_default.json \
  data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200 \
  --variant images --images-2class --num-images 200 --train-ratio 0.8 --copy-mode symlink

EPOCHS=5 make train-ytiny-20250709-taikoB-img2
WEIGHTS=output/train/.../epoch_5.pth make test-ytiny-20250709-taikoB-img2
```

## 注意点
- 分割後の `file_name` はベース名。config 側は data_root + data_prefix.img を train/test ディレクトリに設定済み
- クラス数は COCO既定（80）。2クラスのみで学習したい場合は、画像用のアノテも2クラスに圧縮するコンフィグを別途用意します（要望あれば作成します）
- MPS実行の場合は `scripts/common.sh` の環境変数が反映され、TensorBoard/ ClearML 連携設定は `configs/yolox/yolox_tiny_8xb8-300e_coco_clearml.py` を継承します

## トラブルシュート
- 画像が見つからない
  - file_name がベース名のため、`data_prefix.img` が train/test ディレクトリを指している必要があります
- クラス数不一致エラー
  - 本コンフィグは `num_classes=80`。2クラスにしたい場合は別コンフィグを作成してください
- 評価が落ちる
  - `ann_file` や `data_prefix.img` のパス、`instances_test.json` の整合（CocoMetricが参照）を確認
