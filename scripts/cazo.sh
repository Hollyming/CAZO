#!/bin/bash
PROJECT_ROOT=/home/zjm/Workspace/CAZO
# 切换到项目根目录
cd ${PROJECT_ROOT}
#--tag "_bs${batch_size}_lr${lr}_pertub${pertub}_adapter_layer${adapter_layer}_reduction_factor${reduction_factor}_${adapter_style}_epsilon${epsilon}_nu${nu}" 
#seeds=(42 2020 2025 1234 888) 
export CUDA_VISIBLE_DEVICES=6
seed=42
batch_size=64
lr=0.01
pertub=20
adapter_layer=3
reduction_factor=48 # 48 for swin_tiny, 384 for vit_base
adapter_style="parallel"
fitness_lambda=0.4  # fitness function balance factor

# Hessian related parameters
epsilon=0.1  # Hessian estimation perturbation size
nu=0.8  # Hessian diagonal estimation matrix decay factor (1-nu)D_{t-1} + nu * grad_estimate^2

# optimizer related parameters
optimizer="sgd"  
beta=0.9
continue_learning=False  #--continue_learning ${continue_learning} \
arch=swin_tiny  #vit_base\deit_base\swin_tiny\resnet50


python ${PROJECT_ROOT}/main.py \
    --batch_size ${batch_size} \
    --workers 16 \
    --seed ${seed} \
    --arch ${arch} \
    --data /home/DATA/imagenet \
    --data_v2 /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenetv2 \
    --data_sketch /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-sketch/sketch \
    --data_corruption /home/DATA/imagenet-c \
    --data_rendition /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-r/imagenet-r \
    --dataset_style "imagenet_c" \
    --output ${PROJECT_ROOT}/outputs_new/test/cazo_dit \
    --root_log_dir ${PROJECT_ROOT}/logs_new/test/cazo_dit \
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
    --tag "_seed${seed}_bs${batch_size}_nu${nu}_test"