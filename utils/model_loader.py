# dancher_tools/utils/model_loader.py

import importlib
import sys
import os
import torch

def get_model(args, device):
    model_name = args.model_name
    task_type = args.task

    # 任务类型到模块路径和基类映射
    task_to_paths = {
        'segmentation': {
            'base_module_path': 'dancher_tools.tasks.segmentation',
            'base_class_name': 'SegModel',
        },
        'regression': {
            'base_module_path': 'dancher_tools.tasks.regression',
            'base_class_name': 'RegressionModel',
        },
        'classification': {
            'base_module_path': 'dancher_tools.tasks.classification',
            'base_class_name': 'ClassificationModel',
        }
    }

    if task_type not in task_to_paths:
        raise ValueError(f"Unsupported task type: {task_type}. Supported types are {list(task_to_paths.keys())}")

    # 获取模块路径和基类名称
    paths = task_to_paths[task_type]
    base_module_path = paths['base_module_path']
    base_class_name = paths['base_class_name']

    # 加载基类和参数定义
    try:
        base_module = importlib.import_module(f"{base_module_path}.base")
        base_class = getattr(base_module, base_class_name)
        
        params_module = importlib.import_module(f"{base_module_path}.params")
        required_parameters = getattr(params_module, 'required_parameters', {})
        optional_parameters = getattr(params_module, 'optional_parameters', {})
    except ImportError as e:
        raise ImportError(f"Failed to import base model class or parameters for task '{task_type}': {e}")

    # 验证和组装参数
    task_params = {}
    for param, param_type in required_parameters.items():
        if not hasattr(args, param):
            raise ValueError(f"Missing required parameter '{param}' for task '{task_type}'")
        task_params[param] = param_type(getattr(args, param))
    for param, default_value in optional_parameters.items():
        task_params[param] = getattr(args, param, default_value)

    # 尝试导入预设模型
    try:
        models_module = importlib.import_module(f"{base_module_path}.models")
        preset_models = getattr(models_module, 'PRESET_MODELS', {})
    except ImportError as e:
        raise ImportError(f"Failed to import preset models for task '{task_type}': {e}")

    # 加载并实例化模型（过滤所需参数）
    try:
        if model_name in preset_models:
            model_class = preset_models[model_name]
            print(f"Loaded model '{model_name}' from presets.")
        else:
            # 加载自定义模型模块
            models_path = os.path.join(os.path.dirname(__file__), '../../models')
            if models_path not in sys.path:
                sys.path.append(models_path)
            custom_model_module = f"models.{model_name}"
            model_module = importlib.import_module(custom_model_module)
            model_class = getattr(model_module, model_name)
            print(f"Loaded custom model '{model_name}' from 'models/{model_name}.py'.")

        # 过滤并传入模型所需的参数
        model_params = {}
        if hasattr(model_class, '__init__'):
            init_params = model_class.__init__.__code__.co_varnames
            for param in task_params:
                if param in init_params:
                    model_params[param] = task_params[param]

        model_instance = model_class(**model_params).to(device)

    except Exception as e:
        raise ValueError(f"Error instantiating model '{model_name}': {e}")

    # 使用组合方式包装模型实例
    class WrappedModel(base_class):
        def __init__(self):
            super(WrappedModel, self).__init__(**{k: v for k, v in task_params.items() if k in base_class.__init__.__code__.co_varnames})
            self.model_instance = model_instance
            self.model_name = model_name

        def forward(self, *args, **kwargs):
            return self.model_instance(*args, **kwargs)

    return WrappedModel().to(device)



def load_weights(model, args):
    """
    根据配置加载或迁移模型权重。
    :param model: 初始化后的模型实例
    :param args: 配置对象
    """
    if args.transfer_weights:  # 如果提供了迁移权重路径
        print(f"Transferring model weights from {args.transfer_weights}")
        model.transfer(specified_path=args.transfer_weights, strict=False)
    else:
        # print("Loading model weights from specified path or defaults")
        model.load(model_dir=args.model_save_dir, mode=args.load_mode, specified_path=args.weights)
