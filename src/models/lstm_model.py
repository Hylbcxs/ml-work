import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, args):
        super(LSTMPredictor, self).__init__()
        
        # 模型参数
        self.hidden_size = args.hidden_size
        self.input_feature = args.input_feature
        self.output_feature = args.output_feature
        self.out_seq_length = args.output_length
        self.num_layers = 3  # 增加LSTM层数
        
        # 编码器LSTM - 多层双向
        self.encoder_lstm = nn.LSTM(
            input_size=args.input_feature,
            hidden_size=args.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True  # 双向LSTM更好地捕捉时间序列特征
        )
        
        # 解码器LSTM - 单向（用于生成未来序列）
        self.decoder_lstm = nn.LSTM(
            input_size=args.input_feature,
            hidden_size=args.hidden_size * 2,  # 因为编码器是双向的
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )
        
        # 状态转换层（将双向编码器状态转换为单向解码器状态）
        self.hidden_transform = nn.Linear(args.hidden_size * 2, args.hidden_size * 2)
        self.cell_transform = nn.Linear(args.hidden_size * 2, args.hidden_size * 2)
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LayerNorm(args.hidden_size),  # 添加层归一化
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.hidden_size, args.output_feature)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 设置forget gate偏置为1（LSTM的常见技巧）
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.)
    
    def forward(self, x, target=None):
        batch_size, seq_length, feature_size = x.size()
        
        # 编码阶段 - 双向LSTM处理输入序列
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder_lstm(x)
        
        # 转换编码器状态为解码器初始状态
        # encoder_hidden: [num_layers*2, batch, hidden_size] -> [num_layers, batch, hidden_size*2]
        encoder_hidden = encoder_hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        encoder_hidden = torch.cat([encoder_hidden[:, 0, :, :], encoder_hidden[:, 1, :, :]], dim=2)
        
        encoder_cell = encoder_cell.view(self.num_layers, 2, batch_size, self.hidden_size)
        encoder_cell = torch.cat([encoder_cell[:, 0, :, :], encoder_cell[:, 1, :, :]], dim=2)
        
        # 应用状态转换
        decoder_hidden = self.hidden_transform(encoder_hidden)
        decoder_cell = self.cell_transform(encoder_cell)
        
        # 解码阶段 - 生成未来序列
        decoder_input = x[:, -1:, :]  # 使用最后一个时间步作为初始输入
        decoder_outputs = []
        
        for t in range(self.out_seq_length):
            # LSTM前向传播
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            
            # 生成当前时间步的输出
            current_output = self.output_projection(decoder_output)
            decoder_outputs.append(current_output)
            
            # 准备下一个时间步的输入
            if self.training and target is not None and t < target.size(1) - 1:
                # 训练时使用teacher forcing（概率性地使用真实值）
                use_teacher_forcing = torch.rand(1).item() < 0.7  # 70%概率使用teacher forcing
                if use_teacher_forcing:
                    decoder_input = target[:, t:t+1, :]
                else:
                    decoder_input = current_output
            else:
                # 测试时或teacher forcing失效时使用模型预测
                decoder_input = current_output
            
            # 处理特征维度不匹配的情况
            if decoder_input.size(-1) != self.input_feature:
                if decoder_input.size(-1) < self.input_feature:
                    # 零填充
                    padding_size = self.input_feature - decoder_input.size(-1)
                    padding = torch.zeros(batch_size, 1, padding_size, 
                                        device=decoder_input.device, dtype=decoder_input.dtype)
                    decoder_input = torch.cat([decoder_input, padding], dim=-1)
                else:
                    # 截断
                    decoder_input = decoder_input[:, :, :self.input_feature]
        
        # 合并所有输出
        output = torch.cat(decoder_outputs, dim=1)
        return output