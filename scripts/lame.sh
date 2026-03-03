#!/bin/bash
PROJECT_ROOT=/path/to/project/CAZO
export CUDA_VISIBLE_DEVICES=5
batch_size=64

python3 ${PROJECT_ROOT}/main.py \
    --batch_size ${batch_size} \
    --workers 16\
    --data /path/to/datasets/imagenet \
    --data_v2 /path/to/datasets/imagenetv2 \
    --data_sketch /path/to/datasets/imagenet-sketch/sketch \
    --data_corruption /path/to/datasets/imagenet-c/imagenet-c \
    --data_rendition /path/to/datasets/imagenet-r/imagenet-r \
    --output ${PROJECT_ROOT}/outputs_new/other_datasets \
    --root_log_dir ${PROJECT_ROOT}/logs_new/other_datasets \
    --algorithm "lame" \
    --tag "_bs${batch_size}_other_datasets"