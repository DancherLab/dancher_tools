import importlib
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
    try:
        # 根据任务类型加载对应的预设模型字典
        if args.type == 'segmentation':
            from ..tasks.segmentation.models import PRESET_MODELS
        elif args.type == 'regression':
            from ..tasks.regression.models import PRESET_MODELS
        elif args.type == 'classification':
            from ..tasks.classification.models import PRESET_MODELS
        else:
            raise ValueError(f"Unsupported task type: {args.type}")

        model_name = args.model_name.lower()

        # 预设模型优先加载
        if model_name in PRESET_MODELS:
            model_class = PRESET_MODELS[model_name]
            model = model_class().to(device)  # 直接实例化，不传入 args
            print(f"Loaded model '{model_name}' from presets.")
            return model

        # 若不在预设中，加载自定义模型
        custom_model_module = f"models.{model_name}"
        model_module = importlib.import_module(custom_model_module)
        model_class = getattr(model_module, model_name.capitalize())
        model = model_class().to(device)  # 直接实例化，不传入 args
        print(f"Loaded custom model '{model_name}'.")

        return model
    except ImportError as e:
        raise ImportError(f"Failed to load model '{args.model_name}' for task '{args.type}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Model '{args.model_name}' not found in custom module. Ensure class name matches file name with proper capitalization.")
    except Exception as ex:
        raise ValueError(f"Error in loading model '{args.model_name}': {ex}")
