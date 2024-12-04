import torch.nn as nn
import torch



class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        
        # 根据 num_classes 判断使用 BCE 或 CrossEntropy
        if num_classes == 1:  # 二分类问题或多标签问题
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:  # 多分类问题
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """输入: 
        - inputs: 模型输出（未经过 softmax 或 sigmoid 的 raw logits）
        - targets: 目标标签，形状与输入相同，或是整数型标签（对于 CrossEntropyLoss）
        输出: 损失值
        """
        return self.loss_fn(inputs, targets)


# 自定义的 Dice 损失函数
class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.sigmoid = num_classes == 1  # 二分类使用 sigmoid，其他多分类使用 softmax

    def forward(self, inputs, targets):
        if self.sigmoid:
            inputs = inputs.sigmoid()  # 对于二分类，使用 sigmoid 激活
        else:
            inputs = torch.softmax(inputs, dim=1)  # 对于多分类，使用 softmax 激活

        # 如果 targets 是一个标签（单通道的），需要转化为 one-hot 编码
        if targets.dim() == 3:  # shape = [batch_size, height, width]
            targets = torch.nn.functional.one_hot(targets.long(), num_classes=self.num_classes)
            targets = targets.permute(0, 3, 1, 2).float()  # 转换成 shape = [batch_size, num_classes, height, width]

        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        total = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1 - dice.mean()


# 自定义的 Focal 损失函数
class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = num_classes == 1  # 二分类使用 sigmoid，其他多分类使用 softmax
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        if self.sigmoid:
            inputs = inputs.sigmoid()  # 对于二分类，使用 sigmoid 激活
        else:
            inputs = torch.softmax(inputs, dim=1)  # 对于多分类，使用 softmax 激活
        
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)  # pt 是预测为正确类别的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# 自定义的 Jaccard 损失函数
class JaccardLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.sigmoid = num_classes == 1  # 二分类使用 sigmoid，其他多分类使用 softmax

    def forward(self, inputs, targets):
        # 如果 targets 是 3D（[batch_size, height, width]），将其扩展为 4D（[batch_size, 1, height, width]）
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)  # 将 targets 扩展为 [batch_size, 1, height, width]
        
        # 如果 targets 是 4D 但没有分类维度（例如 [batch_size, height, width]），
        # 则需要将其转换为 one-hot 编码，假设是二分类或多类任务
        if targets.size(1) == 1:  # 如果目标是单通道（例如 [batch_size, 1, height, width]）
            targets = targets.repeat(1, inputs.size(1), 1, 1)  # 扩展为多通道形式
        
        if self.sigmoid:
            inputs = inputs.sigmoid()  # 对于二分类，使用 sigmoid 激活
        else:
            inputs = torch.softmax(inputs, dim=1)  # 对于多分类，使用 softmax 激活
        
        # 计算交集
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        
        # 计算并集
        union = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        
        # 计算 Jaccard 损失
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - jaccard.mean()


# 分割任务的预设损失函数
PRESET_LOSSES = {
    "ce": CrossEntropyLoss,               # 二元交叉熵损失或多类交叉熵损失
    "dice": DiceLoss,  # Dice 损失（自定义）
    "focal": FocalLoss,  # Focal 损失（自定义）
    "jaccard": JaccardLoss  # Jaccard 损失（自定义）
}
