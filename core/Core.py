import torch
import torch.nn as nn
import os
import glob
from datetime import datetime
import re
from tqdm import tqdm
from dancher_tools.utils import CombinedLoss 

class Core(nn.Module):
    def __init__(self):
        super(Core, self).__init__()
        self.model_name = None  # Default model name
        self.last_epoch = 0  # Initialize last_epoch
        self.best_val = 0
        self.optimizer = None
        self.criterion = None
        self.metrics = []


    def compile(self,  criterion, optimizer=None, metrics=None, loss_weights=None):
        """
        设置模型的优化器、损失函数和评价指标。
        :param optimizer: 优化器实例
        :param criterion: 损失函数实例或损失函数列表
        :param metrics: 指标函数字典
        :param loss_weights: 损失函数对应的权重列表（如果 criterion 是列表）
        """
        self.optimizer = optimizer

        # 设置 criterion
        if isinstance(criterion, list):
            if len(criterion) > 1:
                self.criterion = CombinedLoss(losses=criterion, weights=loss_weights)
            elif len(criterion) == 1:
                self.criterion = criterion[0]  # 单一损失函数
            else:
                raise ValueError("Criterion list cannot be empty.")
        elif callable(criterion):
            self.criterion = criterion
        else:
            raise TypeError("Criterion should be a callable loss function or a list of callable loss functions.")

        # 设置指标函数字典
        self.metrics = {}
        if metrics is not None:
            for metric_name, metric_fn in metrics.items():
                if callable(metric_fn):
                    self.metrics[metric_name] = metric_fn
                else:
                    raise ValueError(f"Metric function '{metric_name}' is not callable.")

        print(f"Model compiled with metrics: {list(self.metrics.keys())}")

        
    def save(self, model_dir='./checkpoints', mode='latest'):
        """
        保存模型至指定目录。
        :param model_dir: 保存的文件夹路径
        :param mode: 保存模式，'latest'、'best' 或 'epoch'。
        """
        os.makedirs(model_dir, exist_ok=True)

        # 模式与文件名映射
        mode_to_filename = {
            'latest': f"{self.model_name}_latest.pth",
            'best': f"{self.model_name}_best.pth",
            'epoch': f"{self.model_name}_epoch_{self.last_epoch}.pth",
        }

        # 确认 mode 合法性
        if mode not in mode_to_filename:
            raise ValueError(f"Invalid mode '{mode}'. Valid options are: {list(mode_to_filename.keys())}")

        save_path = os.path.join(model_dir, mode_to_filename[mode])

        # 确定当前设备
        device = next(self.parameters()).device

        # 保存的状态字典
        save_dict = {
            'epoch': getattr(self, 'last_epoch', 0),
            'model_state_dict': self.state_dict(),  # 不改变当前设备，直接保存
            'best_val': getattr(self, 'best_val', None),
        }

        # # 检查并保存优化器状态（如有）
        # optimizer = getattr(self, 'optimizer', None)
        # if optimizer is not None:
        #     save_dict['optimizer_state_dict'] = optimizer.state_dict()

        # 执行保存，捕获异常
        try:
            torch.save(save_dict, save_path, pickle_protocol=4)  # 高效保存
            print(f"Model successfully saved to {save_path}")
        except Exception as e:
            print(f"Failed to save model to {save_path}. Error: {str(e)}")



    def load(self, model_dir='./checkpoints', mode='latest', specified_path=None):
        """
        从指定路径加载模型。
        :param model_dir: 模型文件的目录
        :param mode: 加载模式，'latest'、'best' 或 'epoch'。
        :param specified_path: 直接指定的模型路径（若提供，将优先于 mode 和 model_dir）
        """
        # 如果指定了路径，直接使用 specified_path，否则根据 mode 生成路径
        if specified_path:
            load_path = specified_path
        else:
            if mode == 'latest':
                load_path = os.path.join(model_dir, f"{self.model_name}_latest.pth")
            elif mode == 'best':
                load_path = os.path.join(model_dir, f"{self.model_name}_best.pth")
            elif mode == 'epoch':
                load_path = os.path.join(model_dir, f"{self.model_name}_epoch_{self.last_epoch}.pth")
            else:
                raise ValueError("Invalid mode. Use 'latest', 'best', or 'epoch'.")

        # 加载模型
        if os.path.exists(load_path):
            print(f"Loading model from {load_path}")
            checkpoint = torch.load(load_path, weights_only=False)

            # 加载模型状态
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model successfully loaded from {load_path}, epoch: {checkpoint.get('epoch', 'unknown')}.")

            # 恢复训练状态
            self.last_epoch = checkpoint.get('epoch', 0)
            self.best_val = checkpoint.get('best_val', 0)
        else:
            print(f"No model found at {load_path}, starting from scratch.")
            self.last_epoch = 0
            self.best_val = 0


    def transfer(self, specified_path, strict=False):
        """
        使用指定路径加载预训练模型参数，并将符合条件的参数转移到当前模型（不改变训练状态）。
        :param specified_path: 指定的权重文件路径。
        :param strict: 是否严格匹配层结构。如果为False，将跳过不匹配的参数。
        """
        # 检查路径是否为 None 或路径是否存在
        if not specified_path:
            raise ValueError("Transfer path is not specified. Please provide a valid path for transferring weights.")
        if not os.path.exists(specified_path):
            raise FileNotFoundError(f"The specified transfer path does not exist: {specified_path}")

        print(f"Transferring model parameters from {specified_path}")
        
        # 加载检查点
        checkpoint = torch.load(specified_path, weights_only=False)
        checkpoint_state_dict = checkpoint.get('model_state_dict', checkpoint)
        model_state_dict = self.state_dict()

        new_state_dict = {}
        missing_parameters = []
        extra_parameters = []

        # 匹配模型参数
        for name, parameter in model_state_dict.items():
            if name in checkpoint_state_dict:
                if checkpoint_state_dict[name].size() == parameter.size():
                    new_state_dict[name] = checkpoint_state_dict[name]
                else:
                    extra_parameters.append(name)
            else:
                missing_parameters.append(name)

        # 更新当前模型的参数
        self.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully transferred {len(new_state_dict)} parameters from {specified_path}")

        # 输出缺失和多余的参数信息
        if missing_parameters:
            print(f"Parameters not found in checkpoint (using default): {missing_parameters}")
        if extra_parameters:
            print(f"Parameters in checkpoint but not used due to size mismatch: {extra_parameters}")

        print(f"Transfer completed. Matched: {len(new_state_dict)}, Missing: {len(missing_parameters)}, Size mismatch: {len(extra_parameters)}.")
