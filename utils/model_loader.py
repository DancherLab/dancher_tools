import importlib
import sys
import os
import torch

def get_model(args, device):
    """
    根据配置参数中的模型名称加载对应模型类。模型实例不依赖外部参数。
    
    参数:
        args: 包含 `model_name` 和 `type` 的配置对象。
        device: 设备（如 'cpu' 或 'cuda'）。
    
    返回:
        指定的模型实例。
    """
    model_name = args.model_name  # 保持原始大小写
    task_type = args.task

    # 任务类型到预设模型模块的映射
    task_to_preset_map = {
        'segmentation': 'dancher_tools.tasks.segmentation.models',
        'regression': 'dancher_tools.tasks.regression.models',
        'classification': 'dancher_tools.tasks.classification.models'
    }

    # 检查任务类型是否受支持
    if task_type not in task_to_preset_map:
        raise ValueError(f"Unsupported task type: {task_type}. Supported types are {list(task_to_preset_map.keys())}")

    # 尝试从预设模型中加载
    try:
        preset_module = importlib.import_module(task_to_preset_map[task_type])
        preset_models = getattr(preset_module, 'PRESET_MODELS', {})
    except ImportError as e:
        raise ImportError(f"Failed to import preset models for task '{task_type}': {e}")
    
    # 优先从预设模型加载
    if model_name in preset_models:
        try:
            model_class = preset_models[model_name]
            model = model_class().to(device)
            print(f"Loaded model '{model_name}' from presets.")
            return model
        except Exception as e:
            raise ValueError(f"Error instantiating model '{model_name}' from presets: {e}")

    # 如果未找到预设模型，尝试加载自定义模型
    try:
        # 动态将 models 文件夹添加到 sys.path 中
        models_path = os.path.join(os.path.dirname(__file__), '../../models')
        if models_path not in sys.path:
            sys.path.append(models_path)
        
        # 导入自定义模型模块，不改变大小写
        custom_model_module = f"models.{model_name}"
        model_module = importlib.import_module(custom_model_module)
        
        # 获取类并实例化
        model_class = getattr(model_module, model_name)
        model = model_class().to(device)
        print(f"Loaded custom model '{model_name}' from 'models/{model_name}.py'.")

        return model
    except ImportError:
        raise ImportError(f"Custom model module '{model_name}' not found in 'models' directory. Make sure the file exists and is accessible.")
    except AttributeError:
        raise AttributeError(f"Class '{model_name}' not found in 'models/{model_name}.py'. Ensure the class name matches the file name with proper capitalization.")
    except Exception as e:
        raise ValueError(f"Error in loading custom model '{model_name}': {e}")
