import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import importlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import hashlib  # 用于生成 MD5 校验

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


def calculate_md5(file_path):
    """
    计算文件的 MD5 值。
    """
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_cache_valid(cache_path, module_path):
    """
    判断缓存文件是否有效，检查 MD5 值是否匹配。
    """
    if not os.path.exists(cache_path):
        return False

    # 加载缓存文件的元数据
    cache_data = np.load(cache_path)
    if 'md5' not in cache_data:
        return False

    # 计算当前模块的 MD5 值
    current_md5 = calculate_md5(module_path)
    cached_md5 = cache_data['md5'].item()  # 从缓存中读取 MD5 值
    # # 输出调试信息
    # print(f"Cached MD5: {cached_md5}")
    # print(f"Current MD5: {current_md5}")


    return current_md5 == cached_md5


def get_dataloaders(args):
    """
    通用数据加载器生成函数，支持缓存逻辑。
    """
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = getattr(args, 'img_size', None)
    num_classes = args.num_classes  # 动态获取类别数量

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

        # 获取数据集模块路径
        module_path = f'datapacks/{dataset_name}.py'

        for train_path in dataset_config.get('train_paths', []):
            train_cache_path = os.path.join(train_path, f"__cache__.npz")
            if is_cache_valid(train_cache_path, module_path):
                # print(f"Loading cached train dataset from {train_cache_path}")
                train_data = np.load(train_cache_path)
                train_dataset = datapack(train_data)
            else:
                train_data = create_cache(datapack, train_path, image_size, num_classes, transform, train_cache_path, module_path)
                train_dataset = datapack(train_data)
            train_datasets.append(train_dataset)

        for test_path in dataset_config.get('test_paths', []):
            test_cache_path = os.path.join(test_path, f"__cache__.npz")
            if is_cache_valid(test_cache_path, module_path):
                # print(f"Loading cached test dataset from {test_cache_path}")
                test_data = np.load(test_cache_path)
                test_dataset = datapack(test_data)
            else:
                test_data = create_cache(datapack, test_path, image_size, num_classes, transform, test_cache_path, module_path)
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


def process_file(filename, images_dir, masks_dir, transform, dataset_class, img_size, num_classes):
    """
    处理单个图像和掩码文件。
    """
    image_path = os.path.join(images_dir, filename)
    mask_path = os.path.join(masks_dir, filename.replace('.jpg', '.png'))

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_size, img_size), Image.BILINEAR)  # 调整图像大小
    image = np.array(image, dtype=np.uint8)  # 确保图像是 uint8 类型

    # 加载掩码
    mask = Image.open(mask_path).convert('L')  # 转换为灰度图
    mask = np.array(mask, dtype=np.uint8)  # 确保掩码是 uint8 类型
    mask = dataset_class.convert_mask(mask, num_classes, img_size)  # 转换掩码为单通道格式

    return image, mask



def create_cache(datapack, path, img_size, num_classes, transform, cache_path, module_path):
    """
    使用多线程加快缓存创建，同时保存 MD5 值。
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
                [img_size] * len(images_files),
                [num_classes] * len(images_files)
            ),
            desc=f"Creating cache for {path}",
            total=len(images_files)
        ))

    # 收集结果
    for image, mask in results:
        # 检查形状是否一致
        if image.shape != (img_size, img_size, 3):
            raise ValueError(f"Inconsistent image shape: {image.shape}. Expected {(img_size, img_size, 3)}.")
        if mask.shape != (img_size, img_size):
            raise ValueError(f"Inconsistent mask shape: {mask.shape}. Expected {(img_size, img_size)}.")
        images.append(image)
        masks.append(mask)

    # 保存缓存和 MD5 值
    module_md5 = calculate_md5(module_path)
    np.savez_compressed(
        cache_path,
        images=np.array(images, dtype=np.uint8),
        masks=np.array(masks, dtype=np.uint8),
        md5=module_md5
    )
    print(f"Cache created at {cache_path}")

    return {'images': np.array(images, dtype=np.uint8), 'masks': np.array(masks, dtype=np.uint8)}
