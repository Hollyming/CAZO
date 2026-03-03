#!/bin/bash

# DeYO Test Script for CAZO Framework
# Destroying Your Object-centric Inductive Biases
PROJECT_ROOT=/path/to/project/CAZO
export CUDA_VISIBLE_DEVICES=1
batch_size=64
lr=0.01

python3 ${PROJECT_ROOT}/main.py \
    --batch_size ${batch_size} \
    --workers 16\
    --data /XXX/ILSVRC2012 \
    --data_v2 /XXX/imagenetv2 \
    --data_sketch /XXX/imagenet-sketch/sketch \
    --data_corruption /XXX/imagenet-c \
    --data_rendition /XXX/imagenet-r/imagenet-r \
    --dataset_style "imagenet_c" \
    --output ${PROJECT_ROOT}/outputs/test \
    --root_log_dir ${PROJECT_ROOT}/logs/test \
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
