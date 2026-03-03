#!/bin/bash
PROJECT_ROOT=/path/to/project/CAZO

# Switch to project root directory
cd ${PROJECT_ROOT}
#     --tag "_bs${batch_size}_lr${lr}_num_prompts${num_prompts}"  
#seeds=(42 2020 2025 1234 888) 
export CUDA_VISIBLE_DEVICES=4
batch_size=64
lr=0.01
num_prompts=3
seed=42
continue_learning=True
# --continue_learning ${continue_learning} \

python3 ${PROJECT_ROOT}/main.py \
    --batch_size ${batch_size} \
    --workers 8 \
    --seed ${seed} \
    --lr ${lr} \
    --num_prompts ${num_prompts} \
    --data /XXX/ILSVRC2012 \
    --data_v2 /XXX/imagenetv2 \
    --data_sketch /XXX/imagenet-sketch/sketch \
    --data_corruption /XXX/imagenet-c \
    --data_rendition /XXX/imagenet-r/imagenet-r \
    --dataset_style "imagenet_c" \
    --output ${PROJECT_ROOT}/outputs/foa \
    --root_log_dir ${PROJECT_ROOT}/logs/foa \
    --algorithm "foa" \
    --tag "_seed${seed}_bs${batch_size}"

