def get_metrics(args):
    """
    根据配置文件中的任务类型和指定的 metrics 名称加载对应的指标函数。
    :param args: 包含任务类型和 metrics 列表的配置对象
    :return: 指标名称到指标函数的字典
    """
    try:
        # 根据任务类型导入对应的预设指标
        if args.task == 'segmentation':
            from ..tasks.segmentation.metrics import PRESET_METRICS
        elif args.task == 'regression':
            from tasks.regression.metrics import PRESET_METRICS
        elif args.task == 'classification':
            from tasks.classification.metrics import PRESET_METRICS
        else:
            raise ValueError(f"Unsupported task type: {args.type}")

        # 确保配置文件中的 metrics 是一个列表
        metrics_list = args.metrics if isinstance(args.metrics, list) else [m.strip() for m in args.metrics.split(',')]

        # 筛选出有效的指标函数
        metrics = {name: PRESET_METRICS[name] for name in metrics_list if name in PRESET_METRICS}

        # 警告用户任何未找到的指标
        missing_metrics = set(metrics_list) - set(metrics.keys())
        if missing_metrics:
            print(f"Warning: Some metrics are missing from presets and will be ignored: {missing_metrics}")

        # 确认返回的 metrics 字典有效
        if not metrics:
            raise ValueError(f"No valid metrics found for task '{args.type}' with names: {metrics_list}")

        # print(f"Loaded metrics: {list(metrics.keys())}")
        return metrics
    except ImportError as e:
        raise ImportError(f"Failed to load metrics for task '{args.type}': {e}")
    except Exception as ex:
        raise ValueError(f"Error in loading metrics: {ex}")
