import torch
import os
from networks.model_loader import get_model_by_name
from utils.config_loader import get_args
from utils.data_loader import get_dataloaders
from torch.nn import BCEWithLogitsLoss

def main():
    # 解析参数
    args = get_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型
    model = get_model_by_name(args, device)

    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # 加载预训练模型权重
        # 加载预训练模型权重
    if args.weight:
        # 优先通过指定路径加载权重
        if os.path.isfile(args.weight):
            print(f"Loading model weights from {args.weight}")
            model.load(specified_path=args.weight, model_dir=args.model_save_dir, mode=0)
        else:
            print(f"No checkpoint found at {args.weight}")
    elif args.load_mode:
        # 如果未指定权重路径，但指定了加载模式，则通过模式加载权重
        print(f"Loading model weights using load_mode={args.load_mode}")
        model.load(specified_path=None, model_dir=args.model_save_dir, mode=args.load_mode)
    else:
        print("No pre-trained weights loaded, starting from scratch.")

    # 定义损失函数
    criterion = BCEWithLogitsLoss()

    # 加载数据
    _, test_loader = get_dataloaders(args)
    print("Test data loaded.")
    
    # 查看masks的形状
    for images, masks in test_loader:
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")
        break  # 只打印一次即可

    # 假设 data_loader 已经定义
    model.test(data_loader=test_loader, save_dir=args.model_save_dir, export=args.export)


if __name__ == '__main__':
    main()
