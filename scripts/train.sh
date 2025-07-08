# train
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 /opt/data/private/hyl/code/ml-work/src/train.py \
    --mode 'train' \
    --data_path '/opt/data/private/hyl/code/ml-work/data' \
    --epochs 100 \
    --output_size 365 \
    --input_size 90 \
    --hidden_size 128 \
    --input_feature 13 \
    --output_feature 13 \
    --output_length 365 \
    --lr 1e-3 \
    --batch_size 8 \
    --print_every 20 \
    --seed 42 \
    --loss_type 'mae' \
    --log_file '/opt/data/private/hyl/code/ml-work/logs' \
    --save_path '/opt/data/private/hyl/code/ml-work/ckpts' \
    --figure_path '/opt/data/private/hyl/code/ml-work/figures' \