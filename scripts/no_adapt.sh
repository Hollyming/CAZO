#!/bin/bash
PROJECT_ROOT=/path/to/project/CAZO

# basic parameters setting
batch_size=64
workers=16

# No adaptation baseline - no additional parameters needed

export CUDA_VISIBLE_DEVICES=7

python ${PROJECT_ROOT}/main.py \
    --batch_size ${batch_size} \
    --workers ${workers} \
    --data /XXX/ILSVRC2012 \
    --data_v2 /XXX/imagenetv2 \
    --data_sketch /XXX/imagenet-sketch/sketch \
    --data_corruption /XXX/imagenet-c \
    --data_rendition /XXX/imagenet-r/imagenet-r \
    --dataset_style "imagenet_c" \
    --output ${PROJECT_ROOT}/outputs/test \
    --root_log_dir ${PROJECT_ROOT}/logs/test \
    --algorithm "no_adapt" \
    --tag "_bs${batch_size}_test" 