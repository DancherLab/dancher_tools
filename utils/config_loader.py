import argparse
import yaml
import os

# 必需和可选的通用参数字典
required_common_parameters = {
    'task': str,  # 用于确定任务类型
    'model_name': str,
}
optional_common_parameters = {
    'weight': (str, None),
    'load_mode': (str, 'latest'),
    'learning_rate': (float, 0.001),
    'batch_size': (int, 16),
    'num_workers': (int, 4),
    'patience': (int, 10),
    'delta': (float, 0.01),
    'loss': (str, 'bce'),
    "transfer": (bool, False),
    'loss_weights': (str, None),
    'num_epochs': (int, 100),
    'model_save_dir': (str, 'checkpoints'),
    'metrics': (str, None),
    'export': (bool, False),
    'conf': (bool, False),
}

class Config:
    """支持通过属性访问配置参数的配置类"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def load_yaml_config(config_file):
    """加载并解析 YAML 配置文件"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' does not exist.")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def get_task_parameters(task_type):
    """根据任务类型动态加载任务特定参数"""
    try:
        if task_type == 'segmentation':
            from ..tasks.segmentation.params import required_parameters, optional_parameters
        elif task_type == 'regression':
            from ..tasks.regression.params import required_parameters, optional_parameters
        elif task_type == 'classification':
            from ..tasks.classification.params import required_parameters, optional_parameters
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        return required_parameters, optional_parameters
    except ImportError as e:
        raise ImportError(f"Failed to load parameters for task '{task_type}': {e}")

def parse_loss_weights(loss_weights_str):
    """解析逗号分隔的损失权重字符串"""
    try:
        return [float(w) for w in loss_weights_str.split(",")]
    except ValueError:
        raise ValueError("Parameter 'loss_weights' should be a comma-separated list of numbers.")

def parse_metrics(metrics_str):
    """解析逗号分隔的指标字符串"""
    return [metric.strip() for metric in metrics_str.split(',')]

def validate_and_set_defaults(config, task_type):
    """验证配置参数并根据任务类型设置默认值"""
    validated_config = {}
    errors = []

    # 检查必需的通用参数
    for param, param_type in required_common_parameters.items():
        if param in config and isinstance(config[param], param_type):
            validated_config[param] = config[param]
        else:
            errors.append(f"Missing or invalid required parameter '{param}'.")

    # 检查可选的通用参数并设置默认值
    for param, (param_type, default_value) in optional_common_parameters.items():
        if param in config:
            # 特殊处理 loss_weights 和 metrics
            if param == "loss_weights" and isinstance(config[param], str):
                validated_config[param] = parse_loss_weights(config[param])
            elif param == "metrics" and isinstance(config[param], str):
                validated_config[param] = parse_metrics(config[param])
            elif isinstance(config[param], param_type):
                validated_config[param] = config[param]
            else:
                errors.append(f"Parameter '{param}' should be of type {param_type.__name__}.")
        else:
            validated_config[param] = default_value

    # 获取任务特定的参数并验证
    required_params, optional_params = get_task_parameters(task_type)
    for param, param_type in required_params.items():
        if param in config and isinstance(config[param], param_type):
            validated_config[param] = config[param]
        else:
            errors.append(f"Missing or invalid required task-specific parameter '{param}'.")

    for param, (param_type, default_value) in optional_params.items():
        if param in config and isinstance(config[param], param_type):
            validated_config[param] = config[param]
        else:
            validated_config[param] = default_value

    # 处理 `datasets` 字段，支持多数据集配置
    if 'datasets' in config:
        validated_datasets = []
        for ds in config['datasets']:
            if all(k in ds for k in ['name', 'path', 'train', 'test']):
                dataset_config = {
                    'name': ds['name'],
                    'train_paths': [os.path.join(ds['path'], p) for p in ds['train']],
                    'test_paths': [os.path.join(ds['path'], p) for p in ds['test']]
                }
                validated_datasets.append(dataset_config)
            else:
                errors.append("Each dataset entry must contain 'name', 'path', 'train', and 'test' fields.")
        validated_config['datasets'] = validated_datasets
    else:
        errors.append("Missing 'datasets' configuration.")

    # 抛出错误信息
    if errors:
        raise ValueError("Configuration validation errors:\n" + "\n".join(errors))

    # 创建模型保存路径
    os.makedirs(validated_config['model_save_dir'], exist_ok=True)
    return validated_config

def get_config(config_file=None):
    """加载 YAML 配置文件并验证配置参数"""
    if config_file:
        config = load_yaml_config(config_file)
    else:
        parser = argparse.ArgumentParser(description='Load Configuration File')
        parser.add_argument('--config', type=str, required=True, help='Path to the config file')
        args = parser.parse_args()
        config = load_yaml_config(args.config)

    # 确保 'type' 参数存在
    task_type = config.get('task')
    if not task_type:
        raise ValueError("Missing 'task' parameter in config file.")
    
    validated_config = validate_and_set_defaults(config, task_type)
    print(f"Loaded and validated config from {config_file}: {validated_config}")
    
    return Config(validated_config)
