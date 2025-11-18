#!/bin/bash

# DeYO Test Script for CAZO Framework
# Destroying Your Object-centric Inductive Biases
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
    --algorithm "deyo" \
    --tag "_bs${batch_size}_lr${lr}_test"



# DeYO specific parameters
AUG_TYPE="pixel"  # Options: 'occ', 'patch', 'pixel'
PLPD_THRESHOLD=0.2
MARGIN=0.5
MARGIN_E0=0.4

echo "Running DeYO on ImageNet-C..."
echo "Model: $MODEL"
echo "Batch Size: $BATCH_SIZE"
echo "Augmentation Type: $AUG_TYPE"
echo "PLPD Threshold: $PLPD_THRESHOLD"


echo "DeYO testing completed!"
echo "Results saved to: $OUTPUT"
echo "Logs saved to: $LOG_DIR"
