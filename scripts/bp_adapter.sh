#!/bin/bash
# 测试BP_Adapter算法（反向传播，无标签TTA）
PROJECT_ROOT=/home/zjm/Workspace/CAZO
cd ${PROJECT_ROOT}

export CUDA_VISIBLE_DEVICES=4
seed=42
batch_size=64
lr=0.01
adapter_layer=3
reduction_factor=384  # 48 for swin_tiny, 384 for vit_base
adapter_style="parallel"
fitness_lambda=0.4

# BP optimizer parameters
optimizer="sgd"
momentum=0.9
continue_learning=True

arch=vit_base  #vit_base\deit_base\swin_tiny\resnet50,这里无效参数

python ${PROJECT_ROOT}/main_ft.py \
    --batch_size ${batch_size} \
    --workers 16 \
    --seed ${seed} \
    --arch ${arch} \
    --continue_learning ${continue_learning} \
    --data /media/DATA/imagenet-1k/EXTRACTED \
    --data_v2 /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenetv2 \
    --data_sketch /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-sketch/sketch \
    --data_corruption /media/DATA/imagenet-c \
    --data_rendition /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-r/imagenet-r \
    --dataset_style "imagenet_c" \
    --output ${PROJECT_ROOT}/outputs_new/continue_learning/bp_adapter \
    --root_log_dir ${PROJECT_ROOT}/logs_new/continue_learning/bp_adapter \
    --algorithm "bp_adapter" \
    --lr ${lr} \
    --adapter_layer ${adapter_layer} \
    --reduction_factor ${reduction_factor} \
    --adapter_style ${adapter_style} \
    --optimizer ${optimizer} \
    --momentum ${momentum} \
    --fitness_lambda ${fitness_lambda} \
    --tag "_seed${seed}_bs${batch_size}_lr${lr}_bp"
