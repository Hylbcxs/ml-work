#!/bin/bash
# train
export CUDA_VISIBLE_DEVICES=7
PROJECT_ROOT="/opt/data/private/hyl/code/ml-work"
cd $PROJECT_ROOT


# 短期预测
# echo "开始短期预测训练..."
# torchrun --nproc_per_node=1 -m src.training.train_lstm \
#     --mode 'train' \
#     --data_path "${PROJECT_ROOT}/data" \
#     --epochs 100 \
#     --output_size 90 \
#     --input_size 90 \
#     --hidden_size 256 \
#     --input_feature 14 \
#     --output_feature 14 \
#     --output_length 90 \
#     --lr 1e-3 \
#     --batch_size 8 \
#     --print_every 20 \
#     --seed 42 \
#     --loss_type 'mse' \
#     --log_file "${PROJECT_ROOT}/logs" \
#     --save_path "${PROJECT_ROOT}/ckpts" \
#     --figure_path "${PROJECT_ROOT}/figures" \

# 长期预测
echo "开始短期预测训练..."
torchrun --nproc_per_node=1 -m src.training.train_lstm \
    --mode 'train' \
    --data_path "${PROJECT_ROOT}/data" \
    --epochs 100 \
    --output_size 365 \
    --input_size 90 \
    --hidden_size 256 \
    --input_feature 14 \
    --output_feature 14 \
    --output_length 365 \
    --lr 1e-3 \
    --batch_size 8 \
    --print_every 20 \
    --seed 42 \
    --loss_type 'mse' \
    --log_file "${PROJECT_ROOT}/logs" \
    --save_path "${PROJECT_ROOT}/ckpts" \
    --figure_path "${PROJECT_ROOT}/figures" \