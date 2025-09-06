python run_experiment2_separate.py \
    --data_corruption /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-c/imagenet-c \
    --corruption gaussian_noise \
    --severity 5 \
    --batch_size 32 \
    --adapter_layers 3 \
    --reduction_factor 384 \
    --max_batches 100 \
    --gpu 7 \
    --output ./results