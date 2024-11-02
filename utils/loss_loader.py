# dancher_tools/utils/losses.py
import torch.nn as nn
import torch
import importlib


def get_loss(args):
    """根据配置返回损失函数或损失函数列表"""
    loss_names = args.loss.split(',')  # 支持多损失函数，逗号分隔
    loss_weights = getattr(args, 'loss_weights', None)

    # 预设损失函数映射
    preset_losses = {
        "bce": torch.nn.BCEWithLogitsLoss,
        "cross_entropy": torch.nn.CrossEntropyLoss,
        "mse": torch.nn.MSELoss,
        "bce_with_logits": torch.nn.BCEWithLogitsLoss
    }

    # 构建损失函数列表
    losses = []
    for loss_name in loss_names:
        loss_name = loss_name.strip()
        
        # 检查是否为预设损失
        if loss_name in preset_losses:
            loss_class = preset_losses[loss_name]
            losses.append(loss_class())
        else:
            # 尝试从 `losses` 文件夹中导入自定义损失函数
            try:
                module = importlib.import_module(f"losses.{loss_name}")
                loss_class = getattr(module, loss_name)
                losses.append(loss_class())
                print(f"Loaded custom loss function '{loss_name}' from 'losses' folder.")
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to load custom loss '{loss_name}' from 'losses' folder: {e}")


    # 返回单一损失或损失列表
    return losses

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
