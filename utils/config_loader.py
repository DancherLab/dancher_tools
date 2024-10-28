import argparse
import yaml
import os

def load_config(config_file):
    """
    读取 YAML 配置文件，如果文件不存在或无法正确读取，抛出异常。
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' does not exist.")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError(f"Config file '{config_file}' is empty or contains invalid YAML syntax.")
    except yaml.YAMLError as exc:
        raise ValueError(f"Error reading YAML config file '{config_file}': {exc}")
    
    return config

def get_args():
    """
    解析命令行参数和配置文件参数，并设置默认值。
    """
    parser = argparse.ArgumentParser(description='Training Configuration')

    # 配置文件路径
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file')

    # 模型相关参数
    parser.add_argument('--model_name', type=str, help='Name of the model to use')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='ViT patches size')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--n_skip', type=int, default=3, help='Number of skip connections')

    # 数据加载相关参数
    parser.add_argument('--dataset_path', type=str, help='Base path for datasets')
    parser.add_argument('--train_paths', type=str, nargs='+', help='List of training datasets')
    parser.add_argument('--test_paths', type=str, nargs='+', help='List of test datasets')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')

    # 数据增强参数
    parser.add_argument('--augmentations', action='store_true', default=False, help='Whether to use data augmentations')

    # 是否导出输出
    parser.add_argument('--export', action='store_true', default=False, help='Whether to export outputs')
    parser.add_argument('--conf', action='store_true', default=False, help='Whether to export outputs')

    # 训练相关参数
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_save_dir', type=str, default='./checkpoints', help='Directory to save models')

    # EarlyStopping 参数
    parser.add_argument('--patience', type=int, default=10, help='EarlyStopping patience')
    parser.add_argument('--delta', type=float, default=0.01, help='EarlyStopping delta')

    # 预训练模型路径和加载模式
    parser.add_argument('--weight', type=str, help='Path to the pretrained model weights')
    parser.add_argument('--load_mode', type=int, choices=[0, 1, 2, 3],
                        help='Model loading mode: '
                             '0 - Load the latest checkpoint based on epoch number, '
                             '1 - Load the default checkpoint (model.pth), '
                             '2 - Load the latest modified checkpoint based on file modification time, '
                             '3 - Load the best checkpoint (model_best.pth)')

    # 损失函数选择
    parser.add_argument('--loss', type=str, default='bce', help='Choose the loss function to use: mcc_loss or others')
    parser.add_argument('--loss_weights', type=str, default=None, help='Comma-separated list of loss function weights (e.g., "0.7,0.3" for combined losses)')

    # 解析命令行参数
    args, _ = parser.parse_known_args()

    # 加载配置文件
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
            if not hasattr(args, key):
                setattr(args, key, value)

        # 处理 datasets 信息，拼接完整路径
        if 'datasets' in config:
            datasets = config['datasets']
            args.dataset_path = datasets['path']
            args.train_paths = [os.path.join(args.dataset_path, train_path) for train_path in datasets['train']]
            args.test_paths = [os.path.join(args.dataset_path, test_path) for test_path in datasets['test']]

    # 检查如果没有的必需参数，报错
    required_params = ['model_name', 'img_size', 'num_classes', 'train_paths', 'test_paths', 'num_epochs', 'learning_rate', 'model_save_dir']
    for param in required_params:
        if not hasattr(args, param) or getattr(args, param) is None:
            raise ValueError(f"Missing required parameter: {param}")

    print(f"Loaded config: {args}")

    return args
