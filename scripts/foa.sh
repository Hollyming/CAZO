#!/bin/bash
PROJECT_ROOT=/home/zjm/Workspace/CAZO

# 切换到项目根目录
cd ${PROJECT_ROOT}
#     --tag "_bs${batch_size}_lr${lr}_num_prompts${num_prompts}"  
#seeds=(42 2020 2025 1234 888) 
export CUDA_VISIBLE_DEVICES=4
batch_size=64
lr=0.01
num_prompts=3
seed=42
continue_learning=True

python3 ${PROJECT_ROOT}/main.py \
    --batch_size ${batch_size} \
    --workers 8 \
    --seed ${seed} \
    --lr ${lr} \
    --num_prompts ${num_prompts} \
    --continue_learning ${continue_learning} \
    --data /media/DATA/ILSVRC2012 \
    --data_v2 /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenetv2 \
    --data_sketch /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-sketch/sketch \
    --data_corruption /media/DATA/imagenet-c \
    --data_rendition /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-r/imagenet-r \
    --dataset_style "imagenet_c" \
    --output ${PROJECT_ROOT}/outputs_new/continue_learing/foa \
    --root_log_dir ${PROJECT_ROOT}/logs_new/continue_learing/foa \
    --algorithm "foa" \
    --tag "_seed${seed}_bs${batch_size}"

