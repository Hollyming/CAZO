#!/bin/bash

# ETA Test Script for CAZO Framework
# Entropy minimization with Test-time Adaptation (EATA without Fisher)
PROJECT_ROOT=/home/zjm/Workspace/CAZO
export CUDA_VISIBLE_DEVICES=4
batch_size=64
lr=0.01

# algorithm can be: "eta" or "eata"

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
    --algorithm "eata" \
    --tag "_bs${batch_size}_lr${lr}_test"




# ETA specific parameters
MARGIN_E0=0.4
D_MARGIN=0.05

