import torch.nn as nn
import torch
import importlib
import inspect

def get_loss(args):
    """根据配置返回损失函数或损失函数列表"""
    
    # 根据任务类型导入相应任务的预设损失
    if args.task == 'segmentation':
        from ..tasks.segmentation.losses import PRESET_LOSSES
    elif args.task == 'regression':
        from ..tasks.regression.losses import PRESET_LOSSES
    elif args.task == 'classification':
        from ..tasks.classification.losses import PRESET_LOSSES
    else:
        raise ValueError(f"Unsupported task type: {args.task}")

    # 读取损失函数的名称和权重（可选）
    loss_names = args.loss.split(',')
    loss_weights = parse_loss_weights(args.loss_weights)

    # 构建损失函数列表
    losses = []
    for loss_name in loss_names:
        loss_name = loss_name.strip()
        loss_class = PRESET_LOSSES.get(loss_name) or load_custom_loss_class(loss_name)
        
        # 实例化损失函数
        losses.append(create_loss_instance(loss_class, args))

    # 检查是否需要组合损失
    if len(losses) > 1:
        if loss_weights:
            if len(loss_weights) != len(losses):
                raise ValueError("The number of loss weights must match the number of losses.")
            return CombinedLoss(losses, loss_weights)
        else:
            raise ValueError("Multiple losses specified without corresponding weights.")
    
    return losses[0] if len(losses) == 1 else losses


def parse_loss_weights(loss_weights):
    """解析损失函数的权重"""
    if isinstance(loss_weights, str):
        return [float(w) for w in loss_weights.split(',')]
    elif isinstance(loss_weights, list):
        return [float(w) for w in loss_weights]
    return []


def load_custom_loss_class(loss_name):
    """加载自定义损失函数类"""
    try:
        module = importlib.import_module(f"losses.{loss_name}")
        return getattr(module, loss_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to load custom loss '{loss_name}' from 'losses' folder: {e}")


def create_loss_instance(loss_class, args):
    """根据损失函数类和参数实例化损失函数"""
    if is_lambda(loss_class):
        if requires_num_classes_lambda(loss_class):
            return loss_class(num_classes=args.num_classes)
        return loss_class()
    
    if requires_num_classes(loss_class):
        return loss_class(num_classes=args.num_classes)
    
    return loss_class()


def is_lambda(func):
    """检查函数是否为 lambda 函数"""
    return isinstance(func, type(lambda: None))


def requires_num_classes(loss_class):
    """判断损失函数类是否需要 num_classes 参数"""
    init_signature = inspect.signature(loss_class.__init__)
    params = init_signature.parameters
    return 'num_classes' in params


def requires_num_classes_lambda(lambda_func):
    """判断 lambda 函数是否需要 num_classes 参数"""
    try:
        lambda_func(num_classes=5)
        return True
    except TypeError:
        return False


class CombinedLoss(nn.Module):
    """组合多种损失函数"""
    def __init__(self, losses, weights=None):
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
