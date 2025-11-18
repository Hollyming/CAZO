#!/bin/bash

# RoTTA Test Script for CAZO Framework
# Robust Test-Time Adaptation with Category-balanced Memory

# RoTTA specific parameters
ROTTA_NU=0.001
ROTTA_MEMORY_SIZE=64
ROTTA_UPDATE_FREQ=64

PROJECT_ROOT=/home/zjm/Workspace/CAZO
export CUDA_VISIBLE_DEVICES=1
batch_size=64
lr=0.01

python3 ${PROJECT_ROOT}/main.py \
    --batch_size ${batch_size} \
    --workers 16\
    --data /media/DATA/ILSVRC2012 \
    --data_v2 /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenetv2 \
    --data_sketch /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-sketch/sketch \
    --data_corruption /media/DATA/imagenet-c \
    --data_rendition /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-r/imagenet-r \
    --dataset_style "imagenet_c" \
    --output ${PROJECT_ROOT}/outputs_new/test \
    --root_log_dir ${PROJECT_ROOT}/logs_new/test \
    --algorithm "rotta" \
    --tag "_bs${batch_size}_lr${lr}_test"
