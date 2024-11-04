# tasks/regression/losses.py
import torch.nn as nn

# 回归任务的预设损失函数
PRESET_LOSSES = {
    "mse": nn.MSELoss,                 # 均方误差
    "mae": nn.L1Loss,                  # 均绝误差
    "smooth_l1": nn.SmoothL1Loss       # 平滑L1损失（也称Huber Loss）
}
