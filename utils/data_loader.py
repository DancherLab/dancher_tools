import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import hashlib
import cv2
import h5py
from threading import Thread
from scipy.ndimage import zoom
import importlib

# 之前的 DatasetRegistry 保持不变
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

    try:
        with h5py.File(cache_path, 'r') as f:
            cached_md5 = f.attrs.get('md5', None)
            if cached_md5 is None:
                return False
    except Exception as e:
        print(f"Failed to read cache file {cache_path}: {e}")
        return False

    current_md5 = calculate_md5(module_path)

    return current_md5 == cached_md5


def apply_color_map(mask, color_map):
    """
    将掩码根据 color_map 转换为类别索引，使用向量化操作。
    """
    if not isinstance(color_map, dict):
        raise ValueError(f"Invalid color_map: expected a dictionary, got {type(color_map)}.")

    if mask.ndim == 3 and mask.shape[2] == 3:
        # 将 RGB 值编码为单个整数
        mask_encoded = mask[:, :, 0].astype(np.uint32) << 16 | \
                       mask[:, :, 1].astype(np.uint32) << 8 | \
                       mask[:, :, 2].astype(np.uint32)

        # 构建颜色映射表
        color_keys = np.array(list(color_map.keys()), dtype=np.uint8)
        color_values = np.array(list(color_map.values()), dtype=np.int64)
        color_keys_encoded = color_keys[:, 0].astype(np.uint32) << 16 | \
                             color_keys[:, 1].astype(np.uint32) << 8 | \
                             color_keys[:, 2].astype(np.uint32)
        color_to_index = dict(zip(color_keys_encoded, color_values))

        # 应用颜色映射
        indexed_mask = np.vectorize(color_to_index.get)(mask_encoded)
        return indexed_mask
    elif mask.ndim == 2:
        return mask
    else:
        raise ValueError(f"Invalid mask shape: {mask.shape}. Expected (H, W, 3) or (H, W).")

def process_file(args):
    """
    处理单个文件并直接生成完全处理好的数据，使用 OpenCV 加速图像处理。
    """
    filename, images_dir, masks_dir, img_size, color_map = args

    if color_map is None:
        raise ValueError("No color_map provided. Please set a valid color_map for the dataset.")

    image_path = os.path.join(images_dir, filename)
    mask_path = os.path.join(masks_dir, filename.replace('.jpg', '.png'))

    # 使用 OpenCV 读取和调整图像大小
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 读取为 BGR 格式
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0  # 归一化到 [0, 1]

    # 使用 OpenCV 读取和处理掩码
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = apply_color_map(mask, color_map)
    # 调整掩码大小
    zoom_factor = (img_size / mask.shape[0], img_size / mask.shape[1])
    mask = zoom(mask, zoom_factor, order=0)

    return image, mask


def create_cache(datapack, path, img_size, cache_path, module_path):
    images_dir = os.path.join(path, 'images')
    masks_dir = os.path.join(path, 'masks')
    images_files = sorted(os.listdir(images_dir))

    # 动态访问类级别的 color_map 属性
    if hasattr(datapack, 'color_map'):
        color_map = getattr(datapack, 'color_map')
    else:
        raise ValueError(f"The dataset {datapack.__name__} must define a 'color_map' class attribute or method.")

    if not isinstance(color_map, dict):
        raise ValueError("The dataset's color_map must be a dictionary.")

    num_images = len(images_files)
    images_shape = (num_images, img_size, img_size, 3)
    masks_shape = (num_images, img_size, img_size)

    # 提前创建空数组，避免列表追加的开销
    images = np.empty(images_shape, dtype=np.uint8)
    masks = np.empty(masks_shape, dtype=np.uint8)

    # 准备参数列表
    args_list = [
        (filename, images_dir, masks_dir, img_size, color_map)
        for filename in images_files
    ]

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_file, args_list),
            desc=f"Processing images in {path}",
            total=num_images
        ))

    # 收集结果
    for i, (image, mask) in enumerate(results):
        images[i] = (image * 255).astype(np.uint8)  # 如果之前归一化了，这里还原
        masks[i] = mask.astype(np.uint8)

    module_md5 = calculate_md5(module_path)

    # 异步保存数据
    def save():
        with h5py.File(cache_path, 'w') as f:
            f.create_dataset('images', data=images, dtype='uint8')
            f.create_dataset('masks', data=masks, dtype='uint8')
            f.attrs['md5'] = module_md5

    Thread(target=save).start()
    # print(f"Started saving cache to {cache_path} asynchronously.")

    return {'images': images, 'masks': masks}


def get_dataloaders(args):
    """
    通用数据加载器生成函数，支持缓存逻辑。
    """
    batch_size = args.batch_size
    num_workers = args.num_workers
    img_size = args.img_size

    train_datasets, test_datasets = [], []

    # 只加载配置中的单个数据集
    dataset_config = args.ds
    dataset_name = dataset_config['name']
    DatasetRegistry.load_dataset_module(dataset_name)
    datapack = DatasetRegistry.get_dataset(dataset_name)
    module_path = os.path.join('datapacks', f'{dataset_name}.py')

    # 加载训练数据
    for train_path in dataset_config.get('train_paths', []):
        train_cache_path = os.path.join(train_path, "__cache__.h5")
        if is_cache_valid(train_cache_path, module_path):
            with h5py.File(train_cache_path, 'r') as f:
                images = f['images'][:]
                masks = f['masks'][:]
                train_data = {'images': images, 'masks': masks}
        else:
            train_data = create_cache(datapack, train_path, img_size, train_cache_path, module_path)
        train_dataset = datapack(train_data)
        train_datasets.append(train_dataset)

    # 加载测试数据
    for test_path in dataset_config.get('test_paths', []):
        test_cache_path = os.path.join(test_path, "__cache__.h5")
        if is_cache_valid(test_cache_path, module_path):
            with h5py.File(test_cache_path, 'r') as f:
                images = f['images'][:]
                masks = f['masks'][:]
                test_data = {'images': images, 'masks': masks}
        else:
            test_data = create_cache(datapack, test_path, img_size, test_cache_path, module_path)
        test_dataset = datapack(test_data)
        test_datasets.append(test_dataset)

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
