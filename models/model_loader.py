import torch


def get_model(args, device):
    """
    根据配置参数中的模型名称加载对应模型实例。当前仅支持LSTM模型。
    
    参数:
        args: 命令行参数或配置对象，需要包含 `model_name` 属性。
        device: 设备（如 'cpu' 或 'cuda'）。
    
    返回:
        指定的模型实例。
    """
    if args.model_name.lower() == 'lstm':
        from Regression.LSTM import LSTM
        model = LSTM().to(device)
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}. Currently, only 'lstm' is supported.")

    return model
