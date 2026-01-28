#!/bin/bash
# ZO_Base_FT - 使用零阶优化进行有监督微调

PROJECT_ROOT=/home/zjm/Workspace/CAZO
cd ${PROJECT_ROOT}

export CUDA_VISIBLE_DEVICES=5
seed=42
batch_size=64
lr=0.01
adapter_layer=3
reduction_factor=48
adapter_style="parallel"
fitness_lambda=0.4
num_epochs=2

optimizer="sgd"
beta=0.9
pertub=20
epsilon=0.1

arch=vit_base

python ${PROJECT_ROOT}/main_ft.py \
    --batch_size ${batch_size} \
    --workers 16 \
    --seed ${seed} \
    --arch ${arch} \
    --data /media/DATA/imagenet-1k/EXTRACTED \
    --data_v2 /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenetv2 \
    --data_sketch /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-sketch/sketch \
    --data_corruption /media/DATA/imagenet-c \
    --data_rendition /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-r/imagenet-r \
    --dataset_style "imagenet_c" \
    --output ${PROJECT_ROOT}/outputs_new/finetune/zo_base_ft \
    --root_log_dir ${PROJECT_ROOT}/logs_new/finetune/zo_base_ft \
    --algorithm zo_base_ft \
    --lr ${lr} \
    --pertub ${pertub} \
    --epsilon ${epsilon} \
    --adapter_layer ${adapter_layer} \
    --reduction_factor ${reduction_factor} \
    --adapter_style ${adapter_style} \
    --optimizer ${optimizer} \
    --beta ${beta} \
    --fitness_lambda ${fitness_lambda} \
    --num_epochs ${num_epochs} \
    --continue_learning True \
    --tag "_seed${seed}_ft_${num_epochs}epochs"
