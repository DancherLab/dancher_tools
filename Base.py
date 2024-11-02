# dancher_tools/base.py

def base(task_type):
    """
    根据任务类型返回对应的任务模型类。
    """
    if task_type == 'segmentation':
        from .tasks.segmentation.model import SegModel
        return SegModel
    elif task_type == 'classification':
        from tasks.classification.class_model import ClassModel
        return ClassModel
    elif task_type == 'regression':
        from tasks.regression.reg_model import RegModel
        return RegModel
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
