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
    根据给定的参数返回训练和测试数据加载器。
    :param args: 配置参数，包含数据集类型、路径、批量大小等信息。
    :return: 训练和测试数据加载器
    """
    image_size = args.img_size
    batch_size = args.batch_size
    num_workers = args.num_workers

    # 定义图像的基础变换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # 初始化数据集列表
    train_datasets = []
    test_datasets = []

    # 遍历每个数据集的配置
    for dataset_config in args.datasets:
        dataset_name = dataset_config['name']
        
        # 动态加载数据集模块并获取数据集类
        DatasetRegistry.load_dataset_module(dataset_name)
        dataset_class = DatasetRegistry.get_dataset(dataset_name)

        # 初始化当前数据集的训练集
        for train_path in dataset_config['train_paths']:
            train_images_dir = os.path.join(train_path, 'images')
            train_masks_dir = os.path.join(train_path, 'masks')

            if not os.path.exists(train_images_dir) or not os.path.exists(train_masks_dir):
                raise FileNotFoundError(f"Training directory does not exist: {train_images_dir} or {train_masks_dir}")

            train_images = sorted(os.listdir(train_images_dir))
            train_dataset = dataset_class(
                images_dir=train_images_dir,
                masks_dir=train_masks_dir,
                image_filenames=train_images,
                img_size=image_size,
                transform=transform
            )
            train_datasets.append(train_dataset)

        # 初始化当前数据集的测试集
        for test_path in dataset_config['test_paths']:
            test_images_dir = os.path.join(test_path, 'images')
            test_masks_dir = os.path.join(test_path, 'masks')

            if not os.path.exists(test_images_dir) or not os.path.exists(test_masks_dir):
                raise FileNotFoundError(f"Testing directory does not exist: {test_images_dir} or {test_masks_dir}")

            test_images = sorted(os.listdir(test_images_dir))
            test_dataset = dataset_class(
                images_dir=test_images_dir,
                masks_dir=test_masks_dir,
                image_filenames=test_images,
                img_size=image_size,
                transform=transform
            )
            test_datasets.append(test_dataset)

    # 合并所有数据集
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_test_dataset = ConcatDataset(test_datasets)

    # 创建 DataLoader
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        combined_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader
