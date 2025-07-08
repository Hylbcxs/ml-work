# 家庭电力消耗多变量时间序列
# 任务一：使用 LSTM 模型进行预测
## 1. 简介
我们基于过去 90天 的历史数据进行训练，模型将预测未来 90天（短期预测） 和 365天（长期预测） 的总有功功率。
## 2. 环境要求
我们使用uv包进行管理
环境配置：
```
uv venv
source .venv/bin/activate
uv sync
```
## 3. 数据预处理
在运行训练之前，确保已经正确地处理数据。数据预处理脚本位于```src/data_process/data_preprocessing.py```,它负责加载数据并对数据进行必要的清洗、规范化等操作。
## 4. 训练脚本
```
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 ml-work/src/train.py \
    --mode 'train' \
    --data_path 'ml-work/data' \
    --epochs 100 \
    --output_size 90 \
    --input_size 90 \
    --hidden_size 128 \
    --input_feature 13 \
    --output_feature 13 \
    --output_length 90 \
    --lr 1e-3 \
    --batch_size 8 \
    --print_every 10 \
    --seed 42 \
    --loss_type 'mse' \
    --log_file 'ml-work/logs' \
    --save_path 'ml-work/ckpts' \
    --figure_path 'ml-work/figures'
```
## 5. 参数解释
* epochs：训练轮数
* output_size：模型预测的长度, 设置为 90, 即模型预测未来 90 天的功率。
* input_size：模型输入的时间步数, 设置为 90，表示使用过去 90 天的数据进行训练。
* hidden_size：LSTM 网络中的隐藏层大小
* input_feature：输入数据的特征数目
* output_feature: 输出数据的特征数目
* output_length：设置为 90，表示模型每次输出 90 天的预测值。
* print_every：打印日志的频率
* loss_type：损失函数类型，可选mse，mae
## 6. 模型训练与评估
```
cd scripts
bash train.sh
```

