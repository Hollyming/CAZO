#!/bin/bash
PROJECT_ROOT=/path/to/project/CAZO

# basic parameters setting
batch_size=64
workers=16

# SAR specific parameters
# margin_e0=0.4*math.log(1000)  # For ImageNet-C, use 0.4*log(1000) for ImageNet-R use 0.4*log(200)

export CUDA_VISIBLE_DEVICES=7

python ${PROJECT_ROOT}/main.py \
    --batch_size ${batch_size} \
    --workers ${workers} \
    --data /path/to/datasets/imagenet \
    --data_v2 /path/to/datasets/imagenetv2 \
    --data_sketch /path/to/datasets/imagenet-sketch/sketch \
    --data_corruption /path/to/datasets/imagenet-c/imagenet-c \
    --data_rendition /path/to/datasets/imagenet-r/imagenet-r \
    --output ${PROJECT_ROOT}/outputs_new/test \
    --root_log_dir ${PROJECT_ROOT}/logs_new/test \
    --algorithm "sar" \
    --tag "_bs${batch_size}_test" 