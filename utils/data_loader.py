import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

class HWDataLoader(Dataset):
    """
    自定义Dataset，用于从CSV文件加载热浪预测数据。
    """
    def __init__(self, csv_file):
        """
        参数:
            csv_file (str): CSV文件路径。
        """
        self.data = pd.read_csv(csv_file)

        # 定义特征和目标列
        self.features = ['SSTA', 'SLA', 'LAT', 'TEMP_refer', 'DEPTH']
        self.target = 'TEMP_object'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 提取特征和目标值
        feature_values = self.data.loc[idx, self.features].values.astype(float)
        target_value = self.data.loc[idx, self.target].astype(float)

        # 转换为Tensor
        features_tensor = torch.tensor(feature_values, dtype=torch.float32)
        target_tensor = torch.tensor(target_value, dtype=torch.float32)

        return features_tensor, target_tensor

def get_dataloaders(args):
    """
    加载CSV数据并返回训练和测试DataLoader。

    参数:
        args: 包含配置的参数对象，需包含以下属性：
            - train_paths: 训练集CSV文件路径列表。
            - test_paths: 测试集CSV文件路径列表。
            - batch_size: 每批次的样本数。
            - num_workers: DataLoader中的工作线程数。

    返回:
        train_loader: 训练数据的DataLoader。
        test_loader: 测试数据的DataLoader（如果提供）。
    """
    batch_size = args.batch_size
    num_workers = args.num_workers

    # 加载训练数据集
    train_datasets = []
    for train_path in args.train_paths:
        train_dataset = HWDataLoader(csv_file=train_path)
        train_datasets.append(train_dataset)

    # 合并多个训练集
    combined_train_dataset = ConcatDataset(train_datasets)

    # 创建训练集DataLoader
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # 加载测试数据集
    test_loader = None
    if args.test_paths:
        test_datasets = []
        for test_path in args.test_paths:
            test_dataset = HWDataLoader(csv_file=test_path)
            test_datasets.append(test_dataset)
        
        # 合并多个测试集
        combined_test_dataset = ConcatDataset(test_datasets)

        # 创建测试集DataLoader
        test_loader = DataLoader(
            combined_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    return train_loader, test_loader
