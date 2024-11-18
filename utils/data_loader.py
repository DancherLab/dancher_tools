# dancher_tools/utils/data_loader.py

import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import importlib
import numpy as np
from PIL import Image
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor


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
        """
        动态加载数据集模块。
        """
        try:
            importlib.import_module(f'datapacks.{dataset_name}')
            print(f"Successfully loaded dataset module: {dataset_name}")
        except ImportError:
            raise ValueError(f"Dataset type '{dataset_name}' is not recognized.")


def get_dataloaders(args):
    """
    通用数据加载器生成函数，支持缓存逻辑。
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

    # 遍历配置中的每个数据集
    for dataset_config in args.datasets:
        dataset_name = dataset_config['name']

        # 动态加载数据集模块并获取数据集类
        DatasetRegistry.load_dataset_module(dataset_name)
        datapack = DatasetRegistry.get_dataset(dataset_name)

        for train_path in dataset_config.get('train_paths', []):
            train_cache_path = os.path.join(train_path, f"__cache__.npz")
            if os.path.exists(train_cache_path):
                print(f"Loading cached train dataset from {train_cache_path}")
                train_data = np.load(train_cache_path)
                train_dataset = datapack(train_data)
            else:
                # print(f"Creating cache for train dataset at {train_cache_path}")
                train_data = create_cache(datapack, train_path, image_size, transform, train_cache_path)
                train_dataset = datapack(train_data)
            train_datasets.append(train_dataset)

        for test_path in dataset_config.get('test_paths', []):
            test_cache_path = os.path.join(test_path, f"__cache__.npz")
            if os.path.exists(test_cache_path):
                print(f"Loading cached test dataset from {test_cache_path}")
                test_data = np.load(test_cache_path)
                test_dataset = datapack(test_data)
            else:
                # print(f"Creating cache for test dataset at {test_cache_path}")
                test_data = create_cache(datapack, test_path, image_size, transform, test_cache_path)
                test_dataset = datapack(test_data)
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



def process_file(filename, images_dir, masks_dir, transform, dataset_class, img_size):
    """
    单个文件的处理逻辑，支持并行化。
    """
    image_path = os.path.join(images_dir, filename)
    mask_path = os.path.join(masks_dir, filename.replace('.jpg', '.png'))

    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')

    # 图像和掩码处理
    image = transform(image).numpy()
    mask = dataset_class.convert_mask(np.array(mask), img_size)
    return image, mask


def create_cache(datapack, path, img_size, transform, cache_path):
    """
    使用多线程加快缓存创建。
    """
    images_dir = os.path.join(path, 'images')
    masks_dir = os.path.join(path, 'masks')
    images_files = sorted(os.listdir(images_dir))

    images, masks = [], []
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(
                process_file,
                images_files,
                [images_dir] * len(images_files),
                [masks_dir] * len(images_files),
                [transform] * len(images_files),
                [datapack] * len(images_files),
                [img_size] * len(images_files)
            ),
            desc=f"Creating cache for {path}",
            total=len(images_files)
        ))

    # 收集结果
    for image, mask in results:
        images.append(image)
        masks.append(mask)

    # 保存缓存
    np.savez_compressed(cache_path, images=np.array(images), masks=np.array(masks))
    print(f"Cache created at {cache_path}")

    return {'images': np.array(images), 'masks': np.array(masks)}
