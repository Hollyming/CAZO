#!/bin/bash
PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO
export CUDA_VISIBLE_DEVICES=7
batch_size=64

python3 ${PROJECT_ROOT}/main.py \
    --batch_size ${batch_size} \
    --workers 16\
    --data /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet \
    --data_v2 /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenetv2 \
    --data_sketch /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-sketch/sketch \
    --data_corruption /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-c/imagenet-c \
    --data_rendition /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-r/imagenet-r \
    --output ${PROJECT_ROOT}/outputs_new/test \
    --root_log_dir ${PROJECT_ROOT}/logs_new/test \
    --algorithm "t3a" \
    --tag "_bs${batch_size}_test"