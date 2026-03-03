#!/bin/bash
PROJECT_ROOT=/path/to/project/CAZO
cd ${PROJECT_ROOT}
#--tag "_bs${batch_size}_lr${lr}_pertub${pertub}_adapter_layer${adapter_layer}_reduction_factor${reduction_factor}_${adapter_style}_epsilon${epsilon}_nu${nu}" 
#seeds=(42 2020 2025 1234 888) 
export CUDA_VISIBLE_DEVICES=7
seed=42
batch_size=64
lr=0.01
pertub=20
adapter_layer=3
reduction_factor=384
adapter_style="parallel"
fitness_lambda=0.4  # fitness function balance factor

# Hessian related parameters
epsilon=0.1  # Hessian estimation perturbation size
nu=0.8  # Hessian diagonal estimation matrix decay factor (1-nu)D_{t-1} + nu * grad_estimate^2

# optimizer related parameters
optimizer="sgd"  
beta=0.9

continue_learning=false

## if use for CTTA, add below choice
#--continue_learning ${continue_learning} \


python ${PROJECT_ROOT}/main.py \
    --batch_size ${batch_size} \
    --workers 16 \
    --seed ${seed} \
    --data /XXX/ILSVRC2012 \
    --data_v2 /XXX/imagenetv2 \
    --data_sketch /XXX/imagenet-sketch/sketch \
    --data_corruption /XXX/imagenet-c \
    --data_rendition /XXX/imagenet-r/imagenet-r \
    --dataset_style "imagenet_c" \
    --output ${PROJECT_ROOT}/outputs/cazo \
    --root_log_dir ${PROJECT_ROOT}/logs/cazo \
    --algorithm "cazo" \
    --lr ${lr} \
    --pertub ${pertub} \
    --adapter_layer ${adapter_layer} \
    --reduction_factor ${reduction_factor} \
    --adapter_style ${adapter_style} \
    --optimizer ${optimizer} \
    --beta ${beta} \
    --epsilon ${epsilon} \
    --nu ${nu} \
    --fitness_lambda ${fitness_lambda} \
    --tag "_seed${seed}_bs${batch_size}_nu${nu}"