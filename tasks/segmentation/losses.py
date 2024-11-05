# tasks/segmentation/losses.py
import torch.nn as nn

# 分割任务的预设损失函数
PRESET_LOSSES = {
    "bce": nn.BCEWithLogitsLoss,               # 二元交叉熵损失
    "ce": nn.CrossEntropyLoss,       # 多类交叉熵损失
    "dice": lambda: DiceLoss(),                 # Dice 损失（自定义）
    "focal": lambda: FocalLoss()                # Focal 损失（自定义）
}

# 自定义的 Dice 损失函数
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.sigmoid() if inputs.dim() == 4 else inputs
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        total = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1 - dice.mean()

# 自定义的 Focal 损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)  # pt 是预测为正确类别的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
