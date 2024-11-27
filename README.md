# Dancher-Tools

Dancher-Tools 是一个基于 PyTorch 的即插即用深度学习框架，旨在简化深度学习模型的开发和训练过程。无论您从事图像分割、回归还是分类任务，只需自定义配置文件、模型和数据集，即可快速构建高效的深度学习模型。

## 项目文件目录

要使用 Dancher-Tools，您的项目文件目录需要组织如下：

```
project/
├── configs/           # 配置文件目录，存放 .yaml 配置文件
│   └── sample.yaml    # 示例配置文件
├── dancher_tools/     # Dancher-Tools 框架代码
├── datapacks/         # 数据集设置代码目录，存放自定义数据集处理的 .py 文件
│   └── my_dataset.py  # 示例数据集处理代码
├── datasets/          # 数据集目录，存放原始数据集文件
│   └── my_dataset/    # 示例数据集目录
├── models/            # 自定义模型目录，存放 .py 文件
│   └── MyModel.py     # 示例自定义模型文件
├── train.py           # 训练脚本
├── test.py            # 测试脚本
```

### 运行说明

- **训练模型**：
  使用以下命令，通过指定配置文件来启动训练：
  ```bash
  python train.py --configs configs/sample.yaml
  ```

- **测试模型**：
  使用以下命令，通过指定配置文件来启动测试：
  ```bash
  python test.py --configs configs/sample.yaml
  ```

通过这种目录结构和命令，您可以快速启动模型的训练与测试。

## 快速开始

### 1. 创建项目文件夹

进入您需要创建项目的路径并创建文件夹：

```bash
mkdir project_name
cd project_name
```

### 2. 克隆 Dancher-Tools 代码

在项目文件夹下，克隆 Dancher-Tools 代码：

```bash
git clone https://github.com/DancherLab/dancher_tools.git
```

### 3. 安装依赖

进入项目目录并安装所需的 Python 包：

```bash
cd dancher_tools
pip install -r requirements.txt
```

### 4. 自定义配置文件、模型和数据集

- **`configs/`**：存放您的配置文件。例如 `configs/sample.yaml`。
- **`datapacks/`**：存放数据集设置代码，例如 `my_dataset.py`，用于定义数据的处理逻辑。
- **`datasets/`**：存放实际的数据集内容，例如 `my_dataset/` 包含 `images` 和 `masks`。
- **`models/`**：存放自定义的模型代码，例如 `MyModel.py`。

以下是详细说明。


## 配置文件

Dancher-Tools 使用 YAML 配置文件来设置模型、数据集和训练参数，使整个流程更为清晰和可控。

### 示例配置文件

以下是一个示例配置文件 `configs/FLaTO/conf.yaml`：

```yaml
# configs/FLaTO/conf.yaml

# 模型配置
model_name: 'FLaTO'
task: 'segmentation'
img_size: 224
num_classes: 2

# 数据集配置
datasets:
  - name: 'iw_dataset'
    path: 'datasets/IwDA'
    train:
      - 'train'
    test:
      - 'test'

# 训练配置
batch_size: 16
num_epochs: 500
learning_rate: 0.001
model_save_dir: './results/IwDA/TWIC-Net'

# EarlyStopping 配置
patience: 50
delta: 0.005

# 权重加载模式
load_mode: 'best'

# 损失函数配置
loss: 'ce'

# 评估指标
metrics: 'mIoU,precision,recall,f1_score'
```

### 配置参数说明

#### 模型配置

- `model_name`：模型名称，对应您在 `models` 目录下的模型文件（无需文件扩展名）。对于预设的模型，可以选择以下名称：
  - **图像分割（segmentation）任务预设模型**：
    - [UNet](https://github.com/DancherLab/dancher_tools/blob/main/tasks/segmentation/models/UNet.py)
  - **回归（regression）任务预设模型**：
    - [lstm](https://github.com/DancherLab/dancher_tools/blob/main/tasks/regression/models/LSTM.py)
    - [transformer](https://github.com/DancherLab/dancher_tools/blob/main/tasks/regression/models/transformer.py)

- `task`：任务类型，支持：
  - `segmentation`
  - `regression`
  - `classification`

- `img_size`：输入图像的尺寸（宽和高）。

- `num_classes`：分类数量或输出通道数。

#### 数据集配置

- `datasets`：数据集列表，可以包含多个数据集。
  - `name`：数据集名称。
  - `path`：数据集的根目录。
  - `train`：训练数据的子目录列表。
  - `test`：测试数据的子目录列表。

#### 训练配置

- `batch_size`：每个批次的样本数量。

- `num_epochs`：训练的总轮数。

- `learning_rate`：优化器的学习率。

- `model_save_dir`：模型保存的目录。

#### EarlyStopping 配置

- `patience`：早停策略的耐心值。

- `delta`：早停的最小变化。

#### 权重加载模式

- `load_mode`：模型加载模式，支持 `latest`、`best` 和 `epoch`。

#### 损失函数配置

- `loss`：损失函数类型。对于预设的损失函数，可以选择以下名称：
  - **图像分割（segmentation）任务预设损失函数**：
    - [bce](https://github.com/DancherLab/dancher_tools/blob/main/tasks/segmentation/losses.py)
    - [ce](https://github.com/DancherLab/dancher_tools/blob/main/tasks/segmentation/losses.py)
    - [dice](https://github.com/DancherLab/dancher_tools/blob/main/tasks/segmentation/losses.py)
    - [focal](https://github.com/DancherLab/dancher_tools/blob/main/tasks/segmentation/losses.py)
  - **回归（regression）任务预设损失函数**：
    - [mse](https://github.com/DancherLab/dancher_tools/blob/main/tasks/regression/losses.py)
    - [mae](https://github.com/DancherLab/dancher_tools/blob/main/tasks/regression/losses.py)
    - [smooth_l1](https://github.com/DancherLab/dancher_tools/blob/main/tasks/regression/losses.py)

#### 评估指标

- `metrics`：评估模型的指标列表，多个指标用逗号分隔。对于预设的评估指标，可以选择以下名称：
  - **图像分割（segmentation）任务预设指标**：
    - [mIoU](https://github.com/DancherLab/dancher_tools/blob/main/tasks/segmentation/metrics.py)
    - [precision](https://github.com/DancherLab/dancher_tools/blob/main/tasks/segmentation/metrics.py)
    - [recall](https://github.com/DancherLab/dancher_tools/blob/main/tasks/segmentation/metrics.py)
    - [f1_score](https://github.com/DancherLab/dancher_tools/blob/main/tasks/segmentation/metrics.py)
  - **回归（regression）任务预设指标**：
    - [mse](https://github.com/DancherLab/dancher_tools/blob/main/tasks/regression/metrics.py)
    - [rmse](https://github.com/DancherLab/dancher_tools/blob/main/tasks/regression/metrics.py)
    - [mae](https://github.com/DancherLab/dancher_tools/blob/main/tasks/regression/metrics.py)
    - [r2_score](https://github.com/DancherLab/dancher_tools/blob/main/tasks/regression/metrics.py)

## 自定义模型

将您自定义的 PyTorch 模型文件放入 `models` 文件夹下。例如，如果您有一个名为 `FLaTO.py` 的模型文件，请将其复制到 `models` 目录。

**注意**：

- 模型文件应包含一个与文件名同名的类。例如，`FLaTO.py` 文件中应包含一个 `FLaTO` 类。

- 模型类应继承自 `torch.nn.Module`，并实现 `__init__` 和 `forward` 方法。

## 自定义数据集

将您的数据集按照以下结构组织：

```
datasets/
└── IwDA/
    ├── train/
    │   ├── images/
    │   │   ├── img1.jpg
    │   │   └── img2.jpg
    │   └── masks/
    │       ├── img1.png
    │       └── img2.png
    └── test/
        ├── images/
        │   ├── img3.jpg
        │   └── img4.jpg
        └── masks/
            ├── img3.png
            └── img4.png
```

**注意**：

- `images` 文件夹中存放输入图像。
- `masks` 文件夹中存放对应的标签或掩码（针对分割任务）。
- 数据集路径和子目录需要在配置文件中正确指定。

如果您需要自定义数据加载方式，可以在 `datapacks` 文件夹中创建一个新的数据集类，并在其中实现数据加载逻辑。

## 训练模型

使用以下代码来加载配置文件、数据和模型，开始训练：

```python
import torch
import sys
import os
import dancher_tools as dt


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    # 解析参数
    args = dt.utils.get_config()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 获取数据加载器
    train_loader, val_loader = dt.utils.get_dataloaders(args)

    # 初始化模型
    model = dt.utils.get_model(args, device)
    
    metrics = dt.utils.get_metrics(args)

    # 加载或迁移模型权重
    dt.utils.load_weights(model, args)
    
    # 定义损失函数和优化器
    criterion = dt.utils.get_loss(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 判断是否需要使用置信学习来清洗数据
    if args.conf_threshold is not None:
        train_loader = dt.utils.apply_CL(args.conf_threshold, model, train_loader, device)

    # 通过 `compile` 方法配置模型
    model.compile(optimizer=optimizer, criterion=criterion, metrics=metrics, loss_weights=args.loss_weights)

    # 开始训练模型
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        model_save_dir=args.model_save_dir,
        patience=args.patience,
        delta=args.delta,
    )

    # 评估模型性能
    model.evaluate(data_loader=val_loader)

if __name__ == '__main__':
    main()

```

**注意**：

- 确保在 `get_config` 中指定了正确的配置文件路径。
- 训练过程中，模型和日志将保存在配置文件中指定的 `model_save_dir` 中。

## 测试模型

使用以下代码加载训练好的模型并进行测试：

```python
import torch
import sys
import os
import dancher_tools as dt

# 确保当前路径在sys.path中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # 解析参数
    args = dt.utils.get_config()
    # args.weights = 'UNet_best.pth'
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取数据加载器 (测试集)
    _, test_loader = dt.utils.get_dataloaders(args)

    # 初始化模型
    model = dt.utils.get_model(args, device)
    
    # 获取预定义指标
    metrics = dt.utils.get_metrics(args)


    model.load(model_dir=args.model_save_dir, mode=args.load_mode, specified_path=args.weights)

    # 定义损失函数
    criterion = dt.utils.get_loss(args)

    # 配置模型
    model.compile(optimizer=None, criterion=criterion, metrics=metrics, loss_weights=args.loss_weights)

    # 开始评估模型
    test_results = model.evaluate(data_loader=test_loader)


if __name__ == '__main__':
    main()
```

**注意**：

- 确保加载了正确的模型权重（通常是最好的模型 `'best'`）。
- 测试结果将打印在控制台或保存到指定的位置。

## 支持的功能

- **自动化训练与验证**：通过 `fit` 和 `evaluate` 方法，轻松完成模型的训练和评估。
- **自定义模型**：将自定义的模型文件放入 `models` 目录，并在配置文件中指定。
- **自定义数据集**：支持多数据集加载，灵活配置训练和测试数据路径。
- **多任务支持**：支持图像分割、回归和分类等任务类型。
- **早停策略**：内置早停功能，防止过拟合。
- **多指标评估**：支持常用的评价指标，如 mIoU、精确率、召回率、F1 分数等。
- **迁移学习**：支持加载预训练权重，进行迁移学习。
- **数据清洗（可选）**：支持使用置信学习（Confident Learning）进行数据清洗，处理噪声标签。

---

希望以上内容能够帮助您更好地使用 Dancher-Tools 项目。如有任何问题或建议，欢迎在 [GitHub 项目主页](https://github.com/DancherLab/dancher_tools) 提交 issue 或 pull request。

