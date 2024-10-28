import torch
import os
from models.model_loader import get_model
from utils.config_loader import get_args
from utils.data_loader import get_dataloaders
from utils.losses import get_loss_function  # 导入损失函数选择器

def main():
    # 解析参数
    args = get_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(args)

    # 初始化模型
    model = get_model(args, device)
    print(f"Using model name: {args.model_name}")

    # 加载预训练模型权重
    if args.weight and os.path.isfile(args.weight):
        print(f"Loading model weights from {args.weight}")
        model.load(specified_path=args.weight, model_dir=args.model_save_dir, mode=0)
    elif args.load_mode is not None:
        print(f"Loading model weights using load_mode={args.load_mode}")
        model.load(specified_path=None, model_dir=args.model_save_dir, mode=args.load_mode)
    else:
        print("No pre-trained weights loaded, starting from scratch.")

    # 定义损失函数和优化器
    criterion = get_loss_function(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 开始训练
    model.fit(
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=args.num_epochs,
        model_save_dir=args.model_save_dir,
        patience=args.patience,
        delta=args.delta
    )

    # 测试模型
    model.test(data_loader=val_loader)

if __name__ == '__main__':
    main()
