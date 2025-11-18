#!/bin/bash
PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO
cd ${PROJECT_ROOT}

# basic parameters
batch_size=64
lr=0.01
pertub=20
adapter_layer=3
reduction_factor=384
adapter_style="parallel"
fitness_lambda=0.4
epsilon=0.1
optimizer="sgd"
mode="cov_only"
beta=0.9  #if momentum, add beta

# set GPU
export CUDA_VISIBLE_DEVICES=6

# run COZO experiment
python ${PROJECT_ROOT}/main.py \
    --batch_size ${batch_size} \
    --workers 8 \
    --data /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet \
    --data_v2 /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenetv2 \
    --data_sketch /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-sketch/sketch \
    --data_corruption /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-c/imagenet-c \
    --data_rendition /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-r/imagenet-r \
    --output ${PROJECT_ROOT}/outputs_new/quant/quant6_bs64/cozo \
    --root_log_dir ${PROJECT_ROOT}/logs_new/quant/quant6_bs64/cozo \
    --algorithm "cozo" \
    --lr ${lr} \
    --pertub ${pertub} \
    --adapter_layer ${adapter_layer} \
    --reduction_factor ${reduction_factor} \
    --adapter_style ${adapter_style} \
    --optimizer ${optimizer} \
    --beta ${beta} \
    --epsilon ${epsilon} \
    --mode ${mode} \
    --quant \
    --tag "_bs${batch_size}_lr${lr}_pertub${pertub}_adapter_layer${adapter_layer}_reduction_factor${reduction_factor}_${adapter_style}_opt${optimizer}${beta}_quant6" 