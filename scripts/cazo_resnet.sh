#!/bin/bash
PROJECT_ROOT=/home/zjm/Workspace/CAZO
# 切换到项目根目录
cd ${PROJECT_ROOT}
# CAZO-ResNet实验脚本

export CUDA_VISIBLE_DEVICES=7
seed=42
batch_size=64
lr=0.1
pertub=20
# layer1：3 个 block → index 0, 1, 2 输出通道 256
# layer2：4 个 block → index 3, 4, 5, 6 输出通道 512
# layer3：6 个 block → index 7, 8, 9, 10, 11, 12 输出通道 1024
# layer4：3 个 block → index 13, 14, 15 输出通道 2048
# adapter_layer=1
# reduction_factor=64
# adapter_style="parallel"
fitness_lambda=0.4  # fitness function balance factor
architecture="resnet50"

# Hessian related parameters
epsilon=0.1  # Hessian estimation perturbation size
nu=0.8  # Hessian diagonal estimation matrix decay factor (1-nu)D_{t-1} + nu * grad_estimate^2

# optimizer related parameters
optimizer="sgd"  
beta=0.9
continue_learning=True  # 是否继续学习
#     --continue_learning ${continue_learning} \


python ${PROJECT_ROOT}/main.py \
    --batch_size ${batch_size} \
    --workers 16 \
    --seed ${seed} \
    --continue_learning ${continue_learning} \
    --data /home/DATA/imagenet \
    --data_v2 /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenetv2 \
    --data_sketch /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-sketch/sketch \
    --data_corruption /home/DATA/imagenet-c \
    --data_rendition /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-r/imagenet-r \
    --arch ${architecture} \
    --level 5 \
    --dataset_style "imagenet_c" \
    --output ${PROJECT_ROOT}/outputs_new/resnet/cazo_resnet \
    --root_log_dir ${PROJECT_ROOT}/logs_new/resnet/cazo_resnet \
    --algorithm "cazo_resnet" \
    --lr ${lr} \
    --pertub ${pertub} \
    --optimizer ${optimizer} \
    --beta ${beta} \
    --epsilon ${epsilon} \
    --nu ${nu} \
    --fitness_lambda ${fitness_lambda} \
    --tag "_seed${seed}_bs${batch_size}_nu${nu}_lr${lr}_eps${epsilon}"