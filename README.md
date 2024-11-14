# Dancher-Tools

Dancher-Tools 是一个即插即用的深度学习框架，基于 PyTorch 构建，让模型开发更高效、简洁。无论是图像分割、回归还是分类任务，用户只需将模型文件放入 `models` 文件夹，并在配置文件中指定即可完成配置和调用。

## 快速开始

### 1. 获取代码

首先，克隆项目代码：

```
git clone https://github.com/username/dancher_tools.git
```

### 2. 准备模型

将自定义的 PyTorch 模型文件直接放入 `models` 文件夹下。无需将模型继承 `base`，即可在框架中自动检测和调用。例如，假设我们定义了一个名为 `UNet.py` 的分割模型，将其放入 `models` 文件夹中即可。

## 配置 YAML 文件

Dancher-Tools 通过 YAML 文件配置模型和训练参数，确保流程清晰、简洁。以下是一个 YAML 文件（如 `UNet.yaml`）的示例配置：

```
# configs/UNet.yaml

model_name: 'UNet'
task: 'segmentation'
img_size: 224
num_classes: 2
in_channels: 3
datasets:
  - name: 'my_dataset'
    train_paths: ['data/my_dataset/train']
    test_paths: ['data/my_dataset/test']

batch_size: 16
num_epochs: 50
learning_rate: 0.001
model_save_dir: './results/UNet'
patience: 10
delta: 0.005
load_mode: 'best'
loss: 'ce'
metrics: ['mIoU', 'precision', 'recall', 'f1_score']
```

### 参数说明

- **`model_name`**: 指定模型文件名（无需文件扩展名），如 `'UNet'`。
- **`task`**: 任务类型，支持 `'segmentation'`、`'regression'` 和 `'classification'`。
- **`datasets`**: 指定数据集路径，支持多个数据集。
- **`conf`**: 如果为 `True`，训练过程中启用 Confident Learning 数据清洗功能。
- **`metrics`**: 定义的指标列表，支持多种常见的评价指标。

## 训练模型

使用以下代码，加载您的模型、数据和 YAML 配置文件，开始模型的训练和评估。

```python
import torch
import sys
import os
import dancher_tools as dt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # 解析配置参数
    args = dt.utils.get_config()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取数据加载器
    train_loader, val_loader = dt.utils.get_dataloaders(args)

    # 初始化模型
    model = dt.utils.get_model(args, device)

    metrics = dt.utils.get_metrics(args)

    # 加载或迁移模型权重
    if args.transfer:
        if args.weight and os.path.isfile(args.weight):
            print(f"Transferring model weights from {args.weight}")
            model.transfer(specified_path=args.weight, strict=False)
        else:
            raise FileNotFoundError(f"Specified transfer weight file '{args.weight}' not found.")
    else:
        model.load(model_dir=args.model_save_dir, mode=args.load_mode, specified_path=args.weight)

    # 定义损失函数和优化器
    criterion = dt.utils.get_loss(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 配置模型
    model.compile(optimizer=optimizer, criterion=criterion, metrics=metrics, loss_weights=args.loss_weights)

    # 开始训练模型
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        model_save_dir=args.model_save_dir,
        patience=args.patience,
        delta=args.delta,
        conf=args.conf  # 设置为 True 时启用 Confident Learning
    )

    # 评估模型性能
    model.evaluate(data_loader=val_loader)

if __name__ == '__main__':
    main()
```

## 测试模型

使用以下代码加载并测试模型：

```python
import torch
import sys
import os
import dancher_tools as dt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # 解析配置参数
    args = dt.utils.get_config()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取测试数据加载器
    _, test_loader = dt.utils.get_dataloaders(args)

    # 初始化模型
    model = dt.utils.get_model(args, device)
    
    # 获取指定的评价指标
    metrics = dt.utils.get_metrics(args)

    # 加载模型权重
    model.load(model_dir=args.model_save_dir, mode=args.load_mode, specified_path=args.weight)

    # 定义损失函数
    criterion = dt.utils.get_loss(args)

    # 配置模型（不需要优化器，因为是测试阶段）
    model.compile(optimizer=None, criterion=criterion, metrics=metrics, loss_weights=args.loss_weights)

    # 开始评估模型性能
    test_results = model.evaluate(data_loader=test_loader)

if __name__ == '__main__':
    main()
```

## 项目结构

- **`base/`**: 基础模型类，支持不同任务类型。
- **`tasks/`**: 每个任务的功能封装（如 `segmentation`、`regression`、`classification`）。
- **`utils/`**: 包含数据加载器、配置读取、早停和置信学习等实用工具。
- **`configs/`**: 保存 YAML 配置文件的目录。
- **`models/`**: 用户自定义模型目录，只需放入模型文件并在 YAML 中指定文件名即可。

## 支持的功能

- **自动化训练与验证**：通过 `fit` 和 `evaluate` 方法完成模型训练和验证，支持自定义指标。
- **迁移学习**：在 YAML 配置中设置 `transfer` 和 `weight` 来指定迁移学习的权重文件。
- **Confident Learning 数据清洗**：当 `conf=True` 时，启用数据清洗以处理潜在的噪声标签。


