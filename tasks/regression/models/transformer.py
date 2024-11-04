from dancher_tools import base
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, depth_indices):
        """
        将位置编码添加到输入特征上。
        参数:
        - x: [batch_size, sequence_length, hidden_dim]
        - depth_indices: [batch_size, sequence_length]，表示每个位置的 depth 索引
        
        输出:
        - 带有位置编码的输入特征
        """
        position_encodings = self.pe[:, depth_indices, :]  # [batch_size, sequence_length, hidden_dim]
        x = x + position_encodings  # 添加位置编码
        return x


class Transformer(base('regression')):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=4, num_heads=8, max_depth=1000):
        super(Transformer, self).__init__()
        
        # 特征编码层
        self.feature_encoder = nn.Linear(input_dim - 1, hidden_dim)  # depth 不包括在这里
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=max_depth)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 回归输出层
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        输入:
        - x: [batch_size, sequence_length, input_dim]，包含 SSTA, SLA, LAT, TEMP_refer 和 depth
        
        输出:
        - TEMP_object: [batch_size, sequence_length, 1]，每个深度位置的 TEMP_object 值
        """
        # 分离 depth 特征并将其他特征编码
        features = x[:, :, :-1]  # [batch_size, sequence_length, input_dim - 1]
        depth = x[:, :, -1]  # [batch_size, sequence_length]

        # 将 depth 归一化到 [0, max_depth) 范围并转换为整数索引
        depth_indices = torch.clamp((depth / depth.max()) * (self.positional_encoding.pe.size(1) - 1), 0, self.positional_encoding.pe.size(1) - 1).long()

        # 特征编码
        x = self.feature_encoder(features)  # [batch_size, sequence_length, hidden_dim]
        
        # 添加位置编码
        x = self.positional_encoding(x, depth_indices)  # 添加 depth 位置编码
        x = x.permute(1, 0, 2)  # 转置为 [sequence_length, batch_size, hidden_dim]
        
        # Transformer 编码
        x = self.transformer_encoder(x)  # [sequence_length, batch_size, hidden_dim]
        
        # 回归输出
        x = self.regressor(x)  # [sequence_length, batch_size, 1]
        
        return x.permute(1, 0, 2)  # 转换回 [batch_size, sequence_length, 1]
