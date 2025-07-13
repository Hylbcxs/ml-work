import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from typing import Optional

class MultiScaleCNN_TimeSeries(nn.Module):
    def __init__(self, input_feature, hidden_size, kernel_sizes=[3, 7, 15], dropout=0.3):
        super(MultiScaleCNN_TimeSeries, self).__init__()
        
        # 使用多个卷积核来构建多尺度卷积模块
        # 注意：我们将卷积沿着时间步(seq_len)进行应用
        self.conv1 = nn.Conv1d(input_feature, hidden_size, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2)
        self.conv2 = nn.Conv1d(input_feature, hidden_size, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2)
        self.conv3 = nn.Conv1d(input_feature, hidden_size, kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2)

        # Batch normalization 和 Dropout 层
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.residual_proj = nn.Linear(input_feature, hidden_size * 3)

    def forward(self, x):
        residual = self.residual_proj(x)
        # x.shape = [batch_size, seq_len, input_size] --> [batch_size, input_size, seq_len] for Conv1D
        x = x.transpose(1, 2)  # 转换为 [batch_size, input_size, seq_len]

        # 计算每个卷积层的输出
        x1 = F.gelu(self.bn1(self.conv1(x)))  # 卷积操作
        x2 = F.gelu(self.bn2(self.conv2(x)))  
        x3 = F.gelu(self.bn3(self.conv3(x)))  

        # 合并三个尺度的特征图
        x_concat = torch.cat([x1, x2, x3], dim=1)  # 在通道维度拼接

        # 对拼接后的特征应用 Dropout
        x_concat = self.dropout(x_concat)
        x_concat = x_concat.transpose(1, 2)

        output = x_concat + residual
        return self.dropout(output)

class TemporalAttentionModule(nn.Module):
    """模块2: 时间注意力机制模块"""
    
    def __init__(self, input_dim, hidden_size, seq_len):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 位置编码
        self.pos_encoding = self._generate_pos_encoding(seq_len, input_dim)
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, input_dim)
        )
        # 🔧 添加维度投影层：768 → 256
        self.output_projection = nn.Linear(input_dim, hidden_size)
        
    def _generate_pos_encoding(self, seq_len, d_model):
        pos_encoding = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pos_encoding.unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        # 添加位置编码
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # 多头自注意力
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)

        # 🔧 降维投影：768 → 256
        x = self.output_projection(x)
        
        return x

class LSTMGRUFusionModule(nn.Module):
    """模块3: 长短期记忆融合模块"""
    
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        
        # LSTM用于长期依赖
        self.lstm = nn.LSTM(
            input_size, hidden_size, 
            num_layers=2, batch_first=True, 
            dropout=dropout, bidirectional=True
        )
        
        # GRU用于短期模式
        self.gru = nn.GRU(
            input_size, hidden_size,
            num_layers=2, batch_first=True,
            dropout=dropout, bidirectional=True
        )
        
        # 融合权重学习
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 4),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_size * 4, hidden_size)
        
    def forward(self, x):
        # LSTM输出
        lstm_out, _ = self.lstm(x)
        
        # GRU输出
        gru_out, _ = self.gru(x)
        
        # 特征融合
        combined = torch.cat([lstm_out, gru_out], dim=-1)
        
        # 学习融合权重
        fusion_weights = self.fusion_gate(combined)
        
        # 加权融合
        fused = combined * fusion_weights
        
        # 输出投影
        output = self.output_proj(fused)
        
        return output

class FeatureInteractionModule(nn.Module):
    """模块4: 特征交互模块"""
    
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        
        # 自交互层
        self.self_interaction = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 跨时间步交互
        self.temporal_conv = nn.Conv1d(
            input_size, hidden_size,
            kernel_size=5, padding=2
        )
        
        # 特征重要性评估
        self.importance_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 自交互
        attn_out, _ = self.self_interaction(x, x, x)
        
        # 跨时间步交互
        conv_out = self.temporal_conv(attn_out.transpose(1, 2))
        conv_out = conv_out.transpose(1, 2)
        
        # 特征重要性加权
        importance_weights = self.importance_net(conv_out.transpose(1, 2))
        importance_weights = importance_weights.unsqueeze(1)
        
        weighted_features = conv_out * importance_weights
        
        # 残差连接和归一化
        output = self.layer_norm(weighted_features)
        
        return self.dropout(output)

class SimplePredictionHead(nn.Module):
    def __init__(self, input_size, pred_len, output_features=14):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size * 2, pred_len * output_features)
        )
        self.pred_len = pred_len
        self.output_features = output_features
        
    def forward(self, x, target=None):
        # 全局平均池化
        global_feat = torch.mean(x, dim=1)  # [batch, hidden_size]
        
        # 预测
        pred = self.predictor(global_feat)  # [batch, pred_len * output_features]
        pred = pred.view(-1, self.pred_len, self.output_features)
        
        return pred
    """
    改进的序列到序列预测头
    - 支持Teacher Forcing训练
    - 使用双向编码器
    - 添加注意力机制
    - 解决维度匹配问题
    """
    # def __init__(self, input_size, pred_len, output_features, hidden_size=None):
    #     super().__init__()
    #     self.pred_len = pred_len
    #     self.output_features = output_features
    #     self.input_size = input_size
    #     self.hidden_size = hidden_size or input_size
        
    #     # 编码器：双向LSTM获取更好的上下文表示
    #     self.encoder = nn.LSTM(
    #         input_size=input_size,
    #         hidden_size=self.hidden_size,
    #         num_layers=2,
    #         batch_first=True,
    #         dropout=0.1,
    #         bidirectional=True
    #     )
        
    #     # 解码器：单向LSTM生成未来序列
    #     self.decoder = nn.LSTM(
    #         input_size=output_features,  # 解码器输入是目标特征维度
    #         hidden_size=self.hidden_size * 2,  # 匹配双向编码器的输出
    #         num_layers=2,
    #         batch_first=True,
    #         dropout=0.1
    #     )
        
    #     # 状态转换层：将双向编码器状态转为单向解码器状态
    #     self.hidden_transform = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
    #     self.cell_transform = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        
    #     # 输出投影层
    #     self.output_proj = nn.Sequential(
    #         nn.Linear(self.hidden_size * 2, self.hidden_size),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Linear(self.hidden_size, output_features)
    #     )
        
    #     # 输入特征转换层（用于处理维度不匹配）
    #     if input_size != output_features:
    #         self.feature_transform = nn.Linear(input_size, output_features)
    #     else:
    #         self.feature_transform = nn.Identity()
            
    #     # 初始化权重
    #     self._init_weights()
    
    # def _init_weights(self):
    #     """LSTM权重初始化"""
    #     for name, param in self.named_parameters():
    #         if 'weight_ih' in name:
    #             nn.init.xavier_uniform_(param.data)
    #         elif 'weight_hh' in name:
    #             nn.init.orthogonal_(param.data)
    #         elif 'bias' in name:
    #             param.data.fill_(0)
    #             # LSTM forget gate bias 设为1
    #             if 'bias_ih' in name:
    #                 n = param.size(0)
    #                 param.data[n//4:n//2].fill_(1.)
    
    # def forward(self, x, target=None):
    #     """
    #     前向传播
    #     Args:
    #         x: 编码器输入 [batch_size, seq_len, input_size]
    #         target: 解码器目标 [batch_size, pred_len, output_features]（训练时使用）
    #     Returns:
    #         predictions: [batch_size, pred_len, output_features]
    #     """
    #     batch_size = x.size(0)
        
    #     # 1. 编码阶段
    #     encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(x)
        
    #     # 2. 状态转换：双向 -> 单向
    #     # encoder_hidden: [4, batch, hidden_size] -> [2, batch, hidden_size*2]
    #     encoder_hidden = encoder_hidden.view(2, 2, batch_size, self.hidden_size)
    #     encoder_hidden = torch.cat([encoder_hidden[:, 0, :, :], encoder_hidden[:, 1, :, :]], dim=2)
        
    #     encoder_cell = encoder_cell.view(2, 2, batch_size, self.hidden_size)  
    #     encoder_cell = torch.cat([encoder_cell[:, 0, :, :], encoder_cell[:, 1, :, :]], dim=2)
        
    #     # 应用状态转换
    #     decoder_hidden = self.hidden_transform(encoder_hidden)
    #     decoder_cell = self.cell_transform(encoder_cell)
        
    #     # 3. 解码阶段
    #     predictions = []
        
    #     # 初始解码器输入：使用最后一个时间步并转换维度
    #     last_input = x[:, -1:, :]  # [batch_size, 1, input_size]
    #     decoder_input = self.feature_transform(last_input)  # [batch_size, 1, output_features]
        
    #     for t in range(self.pred_len):
    #         # LSTM解码
    #         decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
    #             decoder_input, (decoder_hidden, decoder_cell)
    #         )
            
    #         # 生成预测
    #         pred = self.output_proj(decoder_output)  # [batch_size, 1, output_features]
    #         predictions.append(pred)
            
    #         # 准备下一时间步输入
    #         if self.training and target is not None and t < self.pred_len - 1:
    #             # 训练时：随机使用Teacher Forcing
    #             use_teacher_forcing = torch.rand(1).item() < 0.7
    #             if use_teacher_forcing:
    #                 decoder_input = target[:, t:t+1, :]
    #             else:
    #                 decoder_input = pred.detach()  # 使用预测值，阻断梯度
    #         else:
    #             # 推理时：使用模型预测
    #             decoder_input = pred
        
    #     # 合并所有预测
    #     output = torch.cat(predictions, dim=1)  # [batch_size, pred_len, output_features]
    #     return output

class MTLFS_Model(nn.Module):
    """
    多模块自适应时间序列预测模型 (Multi-Module Adaptive Time Series Forecasting)
    
    模块组成：
    1. 多尺度特征提取模块 (Multi-scale Feature Extraction)
    2. 时间注意力机制模块 (Temporal Attention)
    3. 长短期记忆融合模块 (LSTM-GRU Fusion)
    4. 特征交互模块 (Feature Interaction)
    5. 自适应预测头模块 (Adaptive Prediction Head)
    """
    
    def __init__(self, 
                 input_feature=14, 
                 hidden_size=256, 
                 input_size=90,
                 output_size=90,
                 dropout=0.1):
        super(MTLFS_Model, self).__init__()
        
        self.input_feature = input_feature
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        
        # 模块1: 多尺度特征提取模块
        self.multi_scale_extractor = MultiScaleCNN_TimeSeries(
            input_feature, hidden_size, dropout=dropout
        )
        
        # 模块2: 时间注意力机制模块
        self.temporal_attention = TemporalAttentionModule(
            hidden_size * 3, hidden_size, input_size
        )
        
        # 模块3: 长短期记忆融合模块
        self.memory_fusion = LSTMGRUFusionModule(
            hidden_size, hidden_size, dropout
        )
        
        # 模块4: 特征交互模块
        self.feature_interaction = FeatureInteractionModule(
            hidden_size, hidden_size, dropout
        )
        
        # 使用Seq2Seq预测头
        # self.seq2seq_predictor = Seq2SeqPredictionHead(
        #     input_size=hidden_size,
        #     pred_len=output_size,
        #     output_features=14,
        #     hidden_size=hidden_size//2
        # )
        self.simple_predictor = SimplePredictionHead(input_size=hidden_size,
            pred_len=output_size,
            output_features=14)
        
    def forward(self, x, prediction_type='short'):
        """
        前向传播
        Args:
            x: 输入数据 [batch_size, seq_len, input_size]
            prediction_type: 'short' for 90天预测, 'long' for 365天预测
        """
        # 模块1: 多尺度特征提取
        multi_scale_features = self.multi_scale_extractor(x)
        
        # 模块2: 时间注意力机制
        attended_features = self.temporal_attention(multi_scale_features)
        
        # 模块3: 长短期记忆融合
        memory_features = self.memory_fusion(attended_features)
        
        # 模块4: 特征交互
        interaction_features = self.feature_interaction(memory_features)
        
        # 模块5: 自适应预测
        predictions = self.simple_predictor(interaction_features)
        
        return predictions