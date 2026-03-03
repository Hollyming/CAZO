#!/bin/bash
PROJECT_ROOT=/path/to/project/CAZO

cd ${PROJECT_ROOT}
#--tag "_bs${batch_size}_lr${lr}_pertub${pertub}_adapter_layer${adapter_layer}_epsilon${epsilon}_quant8" 
#seeds=(42 2020 2025 1234 888) 
export CUDA_VISIBLE_DEVICES=7
batch_size=64
lr=0.01
pertub=20
adapter_layer=3
seed=888
reduction_factor=384
adapter_style="parallel"
fitness_lambda=0.4
epsilon=0.1

# optimizer related parameters
optimizer="sgd"
beta=0.9


python ${PROJECT_ROOT}/main.py \
    --batch_size ${batch_size} \
    --workers 16 \
    --seed ${seed} \
    --data /path/to/datasets/imagenet \
    --data_v2 /path/to/datasets/imagenetv2 \
    --data_sketch /path/to/datasets/imagenet-sketch/sketch \
    --data_corruption /path/to/datasets/imagenet-c/imagenet-c \
    --data_rendition /path/to/datasets/imagenet-r/imagenet-r \
    --output ${PROJECT_ROOT}/outputs_new/test \
    --root_log_dir ${PROJECT_ROOT}/logs_new/test \
    --algorithm "zo_base" \
    --lr ${lr} \
    --pertub ${pertub} \
    --adapter_layer ${adapter_layer} \
    --reduction_factor ${reduction_factor} \
    --adapter_style ${adapter_style} \
    --optimizer ${optimizer} \
    --beta ${beta} \
    --epsilon ${epsilon} \
    --tag "_seed${seed}_bs${batch_size}"
    