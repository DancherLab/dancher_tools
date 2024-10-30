import argparse
import yaml
import os

class Config:
    """简单的配置类，支持通过属性访问配置参数"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def load_yaml_config(config_file):
    """读取 YAML 配置文件"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' does not exist.")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    return config

def validate_and_set_defaults(config):
    """验证配置参数的类型，并设置默认值。若缺少必需参数则抛出错误。"""
    validated_config = {}
    errors = []

    # 必需参数列表（没有默认值的参数）
    required_params = {
        'model_name': str,
        'img_size': int,
        'num_classes': int,
        'dataset_path': str,
        'train_paths': list,
        'test_paths': list,
        'model_save_dir': str
    }

    # 默认参数列表（有默认值的参数）
    default_params = {
        'vit_patches_size': 16,
        'n_skip': 3,
        'batch_size': 16,
        'num_workers': 4,
        'color_map': {},
        'augmentations': False,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'patience': 10,
        'delta': 0.01,
        'weight': None,
        'load_mode': 0,
        'loss': 'bce',
        'loss_weights': None
    }

    # 检查必需参数是否存在，且类型匹配
    for param, param_type in required_params.items():
        if param not in config and param != 'train_paths' and param != 'test_paths' and param != 'dataset_path':
            errors.append(f"Missing required parameter '{param}' of type {param_type.__name__}.")
        elif param in config and not isinstance(config[param], param_type):
            actual_type = type(config[param]).__name__
            errors.append(f"Parameter '{param}' should be of type {param_type.__name__}, but got {actual_type}.")

    # 特殊处理 datasets 部分
    if 'datasets' in config:
        datasets = config['datasets']
        
        # 检查 datasets 中的路径信息
        if 'path' in datasets:
            validated_config['dataset_path'] = datasets['path']
        else:
            errors.append("Missing required parameter 'path' in 'datasets'.")

        # 检查 train 和 test 列表
        if 'train' in datasets:
            validated_config['train_paths'] = [os.path.join(validated_config['dataset_path'], path) for path in datasets['train']]
        else:
            errors.append("Missing required parameter 'train' in 'datasets'.")

        if 'test' in datasets:
            validated_config['test_paths'] = [os.path.join(validated_config['dataset_path'], path) for path in datasets['test']]
        else:
            errors.append("Missing required parameter 'test' in 'datasets'.")
    else:
        errors.append("Missing required 'datasets' configuration.")

    # 如果有错误，抛出异常并列出所有错误
    if errors:
        error_message = "Configuration validation errors:\n" + "\n".join(errors)
        raise ValueError(error_message)

    # 对其余必需参数赋值
    for param in required_params:
        if param in config and param not in validated_config:  # 确保没有被 'datasets' 覆盖
            validated_config[param] = config[param]

    # 设置默认值的参数
    for param, default_value in default_params.items():
        validated_config[param] = config.get(param, default_value)

    # 特殊参数类型检查
    if not isinstance(validated_config['train_paths'], list):
        raise TypeError("train_paths should be a list.")
    if not isinstance(validated_config['test_paths'], list):
        raise TypeError("test_paths should be a list.")
    if not isinstance(validated_config['color_map'], dict):
        raise TypeError("color_map should be a dictionary.")

    # 创建模型保存路径
    if not os.path.exists(validated_config['model_save_dir']):
        os.makedirs(validated_config['model_save_dir'])

    return validated_config

def get_config():
    """解析唯一命令行参数 --config，并加载 YAML 配置文件"""
    parser = argparse.ArgumentParser(description='Load Configuration File')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    # 加载 YAML 配置
    config = load_yaml_config(args.config)
    # 验证和设置默认值
    validated_config = validate_and_set_defaults(config)
    print(f"Loaded and validated config from {args.config}: {validated_config}")
    
    # 返回 Config 对象而不是字典
    return Config(validated_config)
