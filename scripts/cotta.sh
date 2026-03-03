#!/bin/bash
PROJECT_ROOT=/path/to/project/CAZO
export CUDA_VISIBLE_DEVICES=7
batch_size=64
lr=0.01
seed=42

python3 ${PROJECT_ROOT}/main.py \
    --batch_size ${batch_size} \
    --workers 16\
    --lr ${lr} \
    --seed ${seed} \
    --data /XXX/ILSVRC2012 \
    --data_v2 /XXX/imagenetv2 \
    --data_sketch /XXX/imagenet-sketch/sketch \
    --data_corruption /XXX/imagenet-c \
    --data_rendition /XXX/imagenet-r/imagenet-r \
    --dataset_style "imagenet_c" \
    --output ${PROJECT_ROOT}/outputs/test \
    --root_log_dir ${PROJECT_ROOT}/logs/test \
    --algorithm "cotta" \
    --tag "_bs${batch_size}_lr${lr}_test"