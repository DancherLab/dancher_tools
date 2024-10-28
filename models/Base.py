import torch
import torch.nn as nn
import os
import glob
from datetime import datetime
import re

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model_name = 'model'  # Default model name
        self.last_epoch = 0  # Initialize last_epoch
        self.best_val = 0

    def load(self, model_dir='./checkpoints', mode=0, specified_path=None, optimizer=None):
        """
        根据 mode 加载预训练模型权重，支持四种模式：
        mode 0: 加载最新 epoch 的模型。
        mode 1: 加载固定名称的模型文件。
        mode 2: 加载最近修改的模型文件。
        mode 3: 加载最佳模型文件（根据验证集性能保存的模型）。
        """
        load_path = self._get_load_path(specified_path, self.model_name, model_dir, mode)
        
        if load_path and os.path.exists(load_path):
            print(f"Loading model from {load_path}")
            checkpoint = torch.load(load_path, weights_only=True)

            # 加载模型参数
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model successfully loaded from {load_path}, epoch: {checkpoint.get('epoch', 'unknown')}.")
            else:
                print(f"Error: No 'model_state_dict' found in checkpoint {load_path}.")

            # 加载优化器状态（如果提供）
            if optimizer and 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer state loaded.")
                except Exception as e:
                    print(f"Failed to load optimizer state: {e}")

            # 更新训练周期数
            self.last_epoch = checkpoint.get('epoch', 0)
            self.best_val = checkpoint.get('best_val', None)  # 加载最佳验证集性能
            print(f"Last epoch loaded: {self.last_epoch}, Best validation value: {self.best_val}")
        else:
            print(f"No model found at {load_path}, starting from scratch.")
            self.last_epoch = 0
            self.best_val = None


    def transfer(self, path, strict=False):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            checkpoint_state_dict = checkpoint['model_state_dict']
            model_state_dict = self.state_dict()

            # Prepare a dictionary to hold parameters that are available in both checkpoint and current model
            new_state_dict = {}

            # Parameters not found in the current model but exist in the checkpoint
            missing_parameters = []
            extra_parameters = []

            for name, parameter in model_state_dict.items():
                if name in checkpoint_state_dict:
                    if checkpoint_state_dict[name].size() == parameter.size():
                        new_state_dict[name] = checkpoint_state_dict[name]
                    else:
                        extra_parameters.append(name)
                else:
                    missing_parameters.append(name)

            # Loading the matched parameters
            self.load_state_dict(new_state_dict, strict=False)
            print(f"Model parameters transferred from {path}. Successfully loaded parameters: {len(new_state_dict)}")

            if missing_parameters:
                print(f"Parameters not found in the checkpoint and using default: {missing_parameters}")
            if extra_parameters:
                print(f"Parameters in checkpoint but not used due to size mismatch: {extra_parameters}")

        else:
            print(f"No checkpoint found at {path} to transfer parameters from.")

    def save(self, epoch, optimizer=None, model_dir='./checkpoints', mode=0, specified_path=None):
        """
        根据不同的 mode 保存模型：
        mode 0: 保存为带有 epoch 编号的模型。
        mode 1: 保存为固定名称的模型文件。
        mode 2: 保存为带有时间戳的模型文件。
        mode 3: 保存为最佳模型（根据验证集性能）。
        如果提供了 specified_path，则按该路径保存模型。
        """
        # 如果指定了路径，直接使用 specified_path，否则根据模式生成路径
        if specified_path:
            save_path = specified_path
        else:
            save_path = self._determine_save_path(model_dir, self.model_name, epoch, mode)

        # 确保保存路径的目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'best_val': self.best_val  # 保存最佳验证集性能
        }

        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")


    def _remove_batchnorm_state(self, checkpoint_model_state_dict):
        """Remove batch normalization layer's runtime state parameters."""
        return {k: v for k, v in checkpoint_model_state_dict.items() if "running_mean" not in k and "running_var" not in k and "num_batches_tracked" not in k}

    def _load_model_parameters(self, optimizer, checkpoint):
        # 移除检查点中批量归一化层的状态字典
        checkpoint_model_state_dict = self._remove_batchnorm_state(checkpoint['model_state_dict'])
        model_state_dict = self.state_dict()
        new_model_state_dict = {}
        # 记录未从检查点加载的参数名称
        skipped_parameters = []

        # 遍历模型的参数，检查是否存在于检查点中，且形状是否匹配
        for name, parameter in self.named_parameters():
            if name in checkpoint_model_state_dict and parameter.size() == checkpoint_model_state_dict[name].size():
                # 如果匹配，则添加到新的状态字典中
                new_model_state_dict[name] = checkpoint_model_state_dict[name]
            else:
                # 如果不匹配，保持模型当前的参数不变，并记录跳过的参数名称
                new_model_state_dict[name] = model_state_dict[name]
                skipped_parameters.append(name)
        
        self.load_state_dict(new_model_state_dict, strict=False)
        if skipped_parameters:
            print("Skipped parameters (not loaded from checkpoint):")
            for param_name in skipped_parameters:
                print(param_name)
        
        # # 保留原来的优化器加载逻辑
        # if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        #     try:
        #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     except Exception as e:
        #         print(f"Failed to load optimizer state: {e}")
        

    def _get_load_path(self, specified_path, model_name, model_dir, mode):
        if specified_path:
            return specified_path
        if mode == 0:
            pattern = os.path.join(model_dir, f'{model_name}_epoch_*.pth')
            files = glob.glob(pattern)
            epochs = [int(re.search('epoch_([0-9]+)', f).group(1)) for f in files]
            if epochs:
                return os.path.join(model_dir, f'{model_name}_epoch_{max(epochs)}.pth')
        elif mode == 1:
            return os.path.join(model_dir, f'{model_name}.pth')
        elif mode == 2:
            files = glob.glob(os.path.join(model_dir, f'{model_name}_*.pth'))
            if files:
                return sorted(files, key=os.path.getmtime)[-1]
        elif mode == 3:
            return os.path.join(model_dir, f'{model_name}_best.pth')
        return None


    def _determine_save_path(self, model_dir, model_name, last_epoch, mode):
        if mode == 0:
            return os.path.join(model_dir, f'{model_name}_epoch_{last_epoch}.pth')
        elif mode == 1:
            return os.path.join(model_dir, f'{model_name}.pth')
        elif mode == 2:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            return os.path.join(model_dir, f'{model_name}_{timestamp}.pth')
        elif mode == 3:
            return os.path.join(model_dir, f'{model_name}_best.pth')
        else:
            raise ValueError(f"Unsupported mode: {mode}")
