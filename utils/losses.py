import torch.nn as nn
import torch

def get_loss_function(args):
    """
    根据配置中的loss参数，返回对应的损失函数。
    """
    # 定义一个字典，将损失函数名称映射到对应的损失函数类
    loss_dict = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss
    }

    # 检查单一损失函数情况
    if args.loss in loss_dict:
        return loss_dict[args.loss]()  # 直接从字典中获取并返回对应的损失函数
    else:
        raise ValueError(f"Unsupported loss function: {args.loss}")


class CombinedLoss(nn.Module):
    """
    组合多种损失函数，用于回归任务的组合损失。
    """
    def __init__(self, losses, weights=None):
        """
        参数:
        - losses: 损失函数列表。
        - weights: 损失函数对应的权重列表，如果为 None，则权重平均分配。
        """
        super(CombinedLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        if weights is None:
            weights = [1.0 / len(losses)] * len(losses)  # 平均权重分配
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, inputs, targets):
        combined_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            combined_loss += weight * loss_fn(inputs, targets)
        return combined_loss
