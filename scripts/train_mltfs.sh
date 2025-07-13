#!/bin/bash

# TMF (MATSF) 模型训练脚本
# 多模块自适应时间序列预测模型

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=5

# 项目根目录
PROJECT_ROOT="/opt/data/private/hyl/code/ml-work"
cd $PROJECT_ROOT

# echo "🚀 开始短期预测训练..."
# torchrun --nproc_per_node=1 -m src.training.train_mtlfs \
#     --mode 'train' \
#     --data_path "${PROJECT_ROOT}/data" \
#     --prediction_mode 'short' \
#     --input_feature 14 \
#     --hidden_size 512 \
#     --input_size 90 \
#     --output_size 90 \
#     --dropout 0.1 \
#     --epochs 1000 \
#     --batch_size 32 \
#     --lr 1e-4 \
#     --loss_type 'mse' \
#     --seed 42 \
#     --print_every 100 \
#     --log_file "${PROJECT_ROOT}/logs" \
#     --save_path "${PROJECT_ROOT}/ckpts" \
#     --figure_path "${PROJECT_ROOT}/figures"


echo "🚀 开始长期预测训练..."
torchrun --nproc_per_node=1 -m src.training.train_mtlfs \
    --mode 'test' \
    --data_path '/opt/data/private/hyl/code/ml-work/data' \
    --input_feature 14 \
    --hidden_size 512 \
    --input_size 90 \
    --output_size 365 \
    --dropout 0.1 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --val_ratio 0.2 \
    --early_stop_patience 20 \
    --val_check_interval 5 \
    --prediction_mode 'long' \
    --loss_type 'mse' \
    --seed 42 \
    --print_every 10 \
    --num_workers 4 \
    --log_file '/opt/data/private/hyl/code/ml-work/logs' \
    --save_path '/opt/data/private/hyl/code/ml-work/ckpts' \
    --figure_path '/opt/data/private/hyl/code/ml-work/figures'

echo "所有训练任务完成！" 

# torchrun --nproc_per_node=1 -m src.training.train_mtlfs_with_validation \
#     --mode 'test' \
#     --data_path '/opt/data/private/hyl/code/ml-work/data' \
#     --input_feature 14 \
#     --hidden_size 256 \
#     --input_size 90 \
#     --output_size 365 \
#     --dropout 0.1 \
#     --epochs 100 \
#     --batch_size 32 \
#     --lr 5e-4 \
#     --weight_decay 1e-4 \
#     --val_ratio 0.2 \
#     --early_stop_patience 20 \
#     --val_check_interval 5 \
#     --prediction_mode 'long' \
#     --loss_type 'mse' \
#     --seed 42 \
#     --print_every 20 \
#     --num_workers 4 \
#     --log_file '/opt/data/private/hyl/code/ml-work/logs' \
#     --save_path '/opt/data/private/hyl/code/ml-work/ckpts' \
#     --figure_path '/opt/data/private/hyl/code/ml-work/figures'

# echo "✅ 训练完成！"