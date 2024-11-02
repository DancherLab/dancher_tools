# Dancher-Tools

Dancher-Tools 是一个即插即用的深度学习框架，基于 PyTorch 构建，让模型开发变得更高效、简洁。无论是图像分割、回归还是分类任务，用户只需将 PyTorch 的 `nn.Module` 替换为 `dt.base('{task_type}')`，即可使用框架的完整功能。

## 快速开始

### 1. 获取代码

首先，克隆项目代码：

```bash
git clone https://github.com/username/dancher_tools.git
```

### 2. 准备模型

将 PyTorch 模型中的 `nn.Module` 替换为 `dt.base('{task_type}')`。例如，对于分割任务，可以这样初始化模型：

```python
import dancher_tools as dt

class MyModel(dt.base('segmentation')):
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义模型结构
```

## 设置配置文件

Dancher-Tools 通过 YAML 文件配置模型和训练参数，确保流程清晰、简洁。以下是一个 YAML 文件（如 `config.yaml`）的示例配置：

```python
# configs/config.yaml

model_name: 'MyModel'
type: 'segmentation'
img_size: 224
num_classes: 1
datasets:
  - name: 'my_dataset'
    path: 'data/my_dataset'
    train:
      - 'train'
    test:
      - 'test'

batch_size: 16
num_epochs: 50
learning_rate: 0.001
model_save_dir: './results/MyModel'
patience: 10
delta: 0.005
load_mode: 'best'
loss: 'bce'
metrics: 'mIoU,precision,recall,f1_score'
```

## 运行训练与评估

使用以下代码，加载您的模型、数据和 YAML 配置文件，开始模型的训练和评估。

```python
import dancher_tools as dt
import torch

# 加载配置
args = dt.utils.get_config('configs/config.yaml')

# 获取数据加载器
train_loader, val_loader = dt.utils.get_dataloaders(args)

# 初始化模型并加载
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = dt.utils.get_model(args, device)
metrics = dt.utils.get_metrics(args)
criterion = dt.utils.get_loss(args)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# 配置模型并开始训练
model.compile(optimizer=optimizer, criterion=criterion, metrics=metrics)
model.fit(train_loader, val_loader, num_classes=args.num_classes, num_epochs=args.num_epochs)

# 评估模型
model.evaluate(val_loader, save_dir=args.model_save_dir)
```

## 项目结构

- **`base/`**: 基础模型类，支持不同任务类型。
- **`tasks/`**: 每个任务的功能封装（如 `segmentation`、`regression`、`classification`）。
- **`utils/`**: 包含数据加载器、配置读取、早停等实用工具。
- **`configs/`**: 保存 YAML 配置文件的目录。

## 主要功能模块

- **`fit`**：执行模型训练。
- **`evaluate`**：评估模型性能，支持自定义指标。
- **`compile`**：配置损失函数、优化器和评价指标。
- **`dataloader`**：加载和预处理数据集。

## 支持迁移学习

Dancher-Tools 提供了简洁的迁移学习接口，通过在 YAML 中设置 `transfer` 参数和指定预训练权重文件路径，可以轻松加载权重进行微调，快速适应新任务。

---

**Dancher-Tools** 提供全面的接口，帮助开发者专注于深度学习模型的设计和优化，欢迎使用并探索更多可能！
