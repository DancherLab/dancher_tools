from dancher_tools import base
import torch.nn as nn


class LSTM(base('regression')):
    def __init__(self):
        super(LSTM, self).__init__()
        # 设置模型的默认参数
        input_size = 1
        hidden_size = 128
        num_layers = 2
        output_size = 1
        
        # 定义网络结构
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # 取最后一个时间步的输出
        return x
