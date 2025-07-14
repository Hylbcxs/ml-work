# 🏠 家庭电力消耗多变量时间序列预测系统

## 🔍 项目简介
本项目基于"Individual Household Electric Power Consumption"数据集，针对家庭总有功功率进行**短期(90天)**与**长期(365天)**时间序列预测研究。采用三种先进模型：

- **LSTM**：捕捉时序依赖的周期性与趋势  
  

- **Transformer**：通过自注意力机制增强长期依赖关系把握  
  

- **MTLFS**：融合多尺度卷积、时间注意力机制和混合记忆模块的创新模型  
  

## ⚙️ 环境配置
我们使用uv包进行管理
```
uv venv
source .venv/bin/activate
uv sync
```
## 📊 数据预处理
### 预处理脚本
```
src/data_process/data_preprocessing.py
```
### 处理流程
1. 数据读取与列名规范化
2. 特殊值处理与日期解析
3. 时间序列完整性处理
4. 缺失值处理（时间感知插值+列均值填充）
5. 按天聚合数据
6. 外部天气数据融合（Open-Meteo API）
7. 特征工程（提取多维时间特征）

### 特征工程结果
| 类型 | 特征数量 | 示例特征 |
|------|------|:----:|
| 电力特征 | 7个 | 有功/无功功率、电压、电流强度等 |
| 气象特征 | 6个 | 降水量、雾天次数等 |
| 温度特征 | 1个 | 日平均温度 |
| 时间特征 | 6个 | 年、月、日、星期几等 |

## 🧠 模型训练与预测
### 🔄 LSTM模型
#### 训练脚本
```
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 ml-work/src/train.py \
    --model_type 'lstm' \
    --mode 'train' \
    --data_path 'ml-work/data' \
    --epochs 100 \
    --output_size 90 \
    --input_size 90 \
    --hidden_size 128 \
    --input_feature 14 \
    --output_feature 14 \
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
#### 关键参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| **hidden_size** | LSTM隐藏层大小 | 128 |
| **teacher_forcing_ratio** | 教师强制比例 | 0.7 |
| **bidirectional** | 使用双向LSTM | True |

### ⚡ Transformer模型
#### 训练脚本
```
python ml-work/src/model/transformer/transformer.py
```
#### 模型优势
- 自注意力机制捕捉长期依赖
- 位置编码保留时序信息
- 自适应输出策略
#### 关键参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| **nhead** | 注意力头数 | 8 |
| **num_layers** | 编码器层数 | 3 |
| **dim_feedforward** | 前馈网络维度 | 512 |
| **output_strategy** | 输出策略(full_sequence/last_step) | full_sequence |

### 🚀 MTLFS改进模型
#### 训练命令
```
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 ml-work/src/train.py \
    --model_type 'mtlfs' \
    --mode 'train' \
    --data_path 'ml-work/data' \
    --epochs 100 \
    --output_size 365 \
    --input_size 90 \
    --hidden_size 512 \
    --input_feature 20 \
    --output_feature 1 \
    --cnn_channels 64 \
    --attn_heads 8 \
    --lr 1e-4 \
    --batch_size 32 \
    --dropout 0.1 \
    --weight_decay 1e-4 \
    --seed 42 \
    --log_file 'ml-work/logs' \
    --save_path 'ml-work/ckpts'
```
#### 创新模块：​​
1. 多尺度卷积(核尺寸3/7/15)
2. 时间注意力机制
3. LSTM-GRU混合记忆模块
4. 特征交互模块

### 📈 评估指标
####  均方误差(MSE)
```
MSE = (1/n) * Σ(y_i - ŷ_i)^2
```
#### 平均绝对误差(MAE)
```
MAE = (1/n) * Σ|y_i - ŷ_i|
```

### 📊 结果对比
| 模型 | 短期预测(MSE) | 长期预测(MSE) | 稳定性(σ) |
|------|---------------|---------------|-----------|
| LSTM | 0.07592 | 0.07735 | 较高 |
| Transformer | 0.00801 | 0.00913 | **最优** |
| MTLFS | 0.01089 | 0.01008 | 中等 |

