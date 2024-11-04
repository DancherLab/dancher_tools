import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import importlib

class DatasetRegistry:
    """
    数据集注册表，用于管理和动态加载不同的数据集。
    """
    _registry = {}

    @classmethod
    def register_dataset(cls, name):
        """
        注册数据集类。
        :param name: 数据集名称
        """
        def wrapper(dataset_class):
            cls._registry[name] = dataset_class
            return dataset_class
        return wrapper

    @classmethod
    def get_dataset(cls, name):
        """
        获取注册的数据集类。
        :param name: 数据集名称
        :return: 注册的数据集类
        """
        dataset_class = cls._registry.get(name)
        if not dataset_class:
            raise ValueError(f"Dataset '{name}' not registered.")
        return dataset_class
    
    @classmethod
    def load_dataset_module(cls, dataset_name):
        try:
            importlib.import_module(f'datapacks.{dataset_name}')
            print(f"Successfully loaded dataset module: {dataset_name}")
        except ImportError:
            raise ValueError(f"Dataset type '{dataset_name}' is not recognized.")

def get_dataloaders(args):
    """
    通用数据加载器生成函数，直接从 `datapacks` 文件夹中动态加载数据集，返回训练和测试 DataLoader。
    
    参数:
        args: 配置参数对象，包含以下属性：
            - task: 任务类型（segmentation, regression, classification 等）
            - datasets: 数据集配置，包含名称、路径等信息
            - batch_size: 批次大小
            - num_workers: 工作线程数
            - img_size: 图像大小（如果适用）
            
    返回:
        train_loader, test_loader: 训练和测试数据加载器。
    """
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = getattr(args, 'img_size', None)

    # 定义图像的基础变换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]) if image_size else None

    train_datasets, test_datasets = [], []

    # 遍历配置中的每个数据集，直接从 `datapacks` 文件夹加载数据集
    for dataset_config in args.datasets:
        dataset_name = dataset_config['name']
        
        # 动态加载数据集模块并获取数据集类
        DatasetRegistry.load_dataset_module(dataset_name)
        dataset_class = DatasetRegistry.get_dataset(dataset_name)

        if args.task == 'regression':
            # 对于 CSV 格式数据集（回归任务）
            for train_path in dataset_config.get('train_paths', []):
                train_dataset = dataset_class(csv_file=train_path)
                train_datasets.append(train_dataset)
            for test_path in dataset_config.get('test_paths', []):
                test_dataset = dataset_class(csv_file=test_path)
                test_datasets.append(test_dataset)

        else:
            # 图像数据集（分割或分类任务）
            for train_path in dataset_config.get('train_paths', []):
                train_images_dir = os.path.join(train_path, 'images')
                train_masks_dir = os.path.join(train_path, 'masks')
                check_directory_exists(train_images_dir, train_masks_dir)

                train_images = sorted(os.listdir(train_images_dir))
                train_dataset = dataset_class(
                    images_dir=train_images_dir,
                    masks_dir=train_masks_dir,
                    image_filenames=train_images,
                    img_size=image_size,
                    transform=transform
                )
                train_datasets.append(train_dataset)

            for test_path in dataset_config.get('test_paths', []):
                test_images_dir = os.path.join(test_path, 'images')
                test_masks_dir = os.path.join(test_path, 'masks')
                check_directory_exists(test_images_dir, test_masks_dir)

                test_images = sorted(os.listdir(test_images_dir))
                test_dataset = dataset_class(
                    images_dir=test_images_dir,
                    masks_dir=test_masks_dir,
                    image_filenames=test_images,
                    img_size=image_size,
                    transform=transform
                )
                test_datasets.append(test_dataset)

    # 合并数据集并创建 DataLoader
    combined_train_dataset = ConcatDataset(train_datasets) if train_datasets else None
    combined_test_dataset = ConcatDataset(test_datasets) if test_datasets else None

    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    ) if combined_train_dataset else None

    test_loader = DataLoader(
        combined_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    ) if combined_test_dataset else None

    return train_loader, test_loader

def check_directory_exists(images_dir, masks_dir):
    """检查图像和掩码文件夹是否存在"""
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Required directory does not exist: {images_dir} or {masks_dir}")
