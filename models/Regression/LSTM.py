import torch.nn as nn
from .Regression import RegressionModel

class LSTM(RegressionModel):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(LSTM, self).__init__()
        self.model_name = 'LSTM_model'
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 如果输入缺少时间步维度，自动增加一个时间维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 增加时间维度 (batch_size, 1, input_size)

        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的输出
        return output
