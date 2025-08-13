.PHONY: help train-yolox-tiny test-yolox-tiny demo-yolox-tiny data-coco setup-links \
	train-yolox-tiny-custom-images test-yolox-tiny-custom-images \
	train-ytiny-20250709-taikoB-img80 test-ytiny-20250709-taikoB-img80 \
	train-ytiny-20250709-taikoB-img2  test-ytiny-20250709-taikoB-img2 \
	splits-20250709-taikoB-img80 splits-20250709-taikoB-img2 \
	test-ytiny-20250709-taikoB-img2-last

help:
	@echo "Available targets:"
	@echo "  make train-yolox-tiny   # 学習(既定1epoch, AMP, ClearML有効)"
	@echo "  make test-yolox-tiny    # 評価(結果pkl保存, ClearML有効)"
	@echo "  make demo-yolox-tiny    # 画像1枚のデモ推論(MPS/env反映)"
	@echo "  make data-coco          # COCO2017 を data/coco にDL+展開"
	@echo "  make setup-links        # USB上のdata/dataset/resultへシンボリックリンク設定"
	@echo "  make test-yolox-tiny-custom-images   # 自前split(test)で学習済み重みを評価"
	@echo "  make train-yolox-tiny-custom-images  # 自前split(train)でYOLOX-Tinyを学習"
	@echo "  make test-ytiny-20250709-taikoB-img80  # 日付/シーン名(img, 80cls)で評価"
	@echo "  make train-ytiny-20250709-taikoB-img80 # 日付/シーン名(img, 80cls)で学習"
	@echo "  make test-ytiny-20250709-taikoB-img2   # 日付/シーン名(img, 2cls)で評価"
	@echo "  make train-ytiny-20250709-taikoB-img2  # 日付/シーン名(img, 2cls)で学習"
	@echo "  make splits-20250709-taikoB-img80      # 分割生成(images, 80cls)"
	@echo "  make splits-20250709-taikoB-img2       # 分割生成(images2, 2cls)"
	@echo "  make test-ytiny-20250709-taikoB-img2-last # 最新epoch重みで2clsテスト"

train-yolox-tiny:
	bash scripts/train_yolox_tiny.sh

test-yolox-tiny:
	bash scripts/test_yolox_tiny.sh

demo-yolox-tiny:
	bash scripts/demo_yolox_tiny.sh

data-coco:
	python tools/misc/download_dataset.py --dataset-name coco2017 --save-dir data/coco --unzip --delete --threads 4

setup-links:
	bash scripts/setup_symlinks.sh

# --- Custom dataset (images variant) ---
test-yolox-tiny-custom-images:
	CONFIG=configs/yolox/yolox_tiny_custom_images_test.py \
	MODEL=yolox_tiny \
	DATASET=ys20250709_images \
	bash scripts/test_yolox_tiny.sh

train-yolox-tiny-custom-images:
	CONFIG=configs/yolox/yolox_tiny_custom_images_train.py \
	MODEL=yolox_tiny \
	DATASET=ys20250709_images \
	EPOCHS=${EPOCHS} \
	bash scripts/train_yolox_tiny.sh

# Short, self-descriptive targets (date/scene/modality/classes)
test-ytiny-20250709-taikoB-img80:
	CONFIG=configs/yolox/ytiny_20250709_toshitaiko_seat-knockB_img80_test.py \
	MODEL=yolox_tiny \
	DATASET=20250709_toshitaiko_seat-knockB_img80 \
	bash scripts/test_yolox_tiny.sh

train-ytiny-20250709-taikoB-img80:
	CONFIG=configs/yolox/ytiny_20250709_toshitaiko_seat-knockB_img80_train.py \
	MODEL=yolox_tiny \
	DATASET=20250709_toshitaiko_seat-knockB_img80 \
	EPOCHS=${EPOCHS} \
	bash scripts/train_yolox_tiny.sh

test-ytiny-20250709-taikoB-img2:
	CONFIG=configs/yolox/ytiny_20250709_toshitaiko_seat-knockB_img2_test.py \
	MODEL=yolox_tiny \
	DATASET=20250709_toshitaiko_seat-knockB_img2 \
	bash scripts/test_yolox_tiny.sh

train-ytiny-20250709-taikoB-img2:
	CONFIG=configs/yolox/ytiny_20250709_toshitaiko_seat-knockB_img2_train.py \
	MODEL=yolox_tiny \
	DATASET=20250709_toshitaiko_seat-knockB_img2 \
	EPOCHS=${EPOCHS} \
	bash scripts/train_yolox_tiny.sh

# Generate splits via pipeline (images 80cls)
splits-20250709-taikoB-img80:
	python tools/misc/build_dataset_pipeline.py \
	  data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/instances_default.json \
	  data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200 \
	  --variant images \
	  --num-images 200 \
	  --train-ratio 0.8 \
	  --copy-mode symlink

# Generate splits via pipeline (images2 2cls)
splits-20250709-taikoB-img2:
	python tools/misc/build_dataset_pipeline.py \
	  data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/instances_default.json \
	  data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200 \
	  --variant images \
	  --images-2class \
	  --num-images 200 \
	  --train-ratio 0.8 \
	  --copy-mode symlink

# Test with the latest produced checkpoint for 2cls
test-ytiny-20250709-taikoB-img2-last:
	CONFIG=configs/yolox/ytiny_20250709_toshitaiko_seat-knockB_img2_test.py \
	MODEL=yolox_tiny \
	DATASET=20250709_toshitaiko_seat-knockB_img2 \
	WEIGHTS=$$(ls -t output/train/20250709_toshitaiko_seat-knockB_img2/yolox_tiny/ytiny_20250709_toshitaiko_seat-knockB_img2_train/*/*/epoch_*.pth | head -n1) \
	bash scripts/test_yolox_tiny.sh
