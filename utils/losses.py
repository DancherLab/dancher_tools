import torch.nn as nn
import torch

class CombinedLoss(nn.Module):
    """
    组合多种损失函数，用于回归任务的组合损失。
    """
    def __init__(self, losses, weights=None):
        """
        参数:
        - losses: 损失函数列表或类列表。
        - weights: 损失函数对应的权重列表，如果为 None，则权重平均分配。
        """
        super(CombinedLoss, self).__init__()

        # 实例化损失函数
        self.losses = nn.ModuleList([loss() if isinstance(loss, type) else loss for loss in losses])
        
        # 设置权重
        if weights is None:
            weights = [1.0 / len(self.losses)] * len(self.losses)
        elif len(weights) != len(self.losses):
            raise ValueError("Length of weights must match the number of loss functions.")
        
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, inputs, targets):
        combined_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            combined_loss += weight * loss_fn(inputs, targets)
        return combined_loss
