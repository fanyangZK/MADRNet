from torch.utils.data import DataLoader
import torch
from model.MADRNet import MADRNet
from utils.transform_config import train_transform,test_transform
from model.HybirdLoss import HybridLoss
from utils.clean_hidden_files import clean_hidden_files
from utils.MetricLogger import MetricLogger
from utils.ImageDataset import ImageDataset
from utils.test_epoch import test_epoch
from utils.train_epoch import train_epoch
import os
class Config:
    data_dir = "data/HuSHeM"  # 替换为实际路径
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    csv_path = "res.csv"
    num_classes = 4
    batch_size = 16
    epochs = 400
    lr_default = 3e-3
    lr_backbone = 1e-3
    lr_rev_blocks = 1e-4
    lr_classifier = 1e-3
    weight_decay = 3e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    # 数据加载与清洗
    clean_hidden_files(Config.train_dir)
    clean_hidden_files(Config.test_dir)

    train_dataset = ImageDataset(Config.train_dir, train_transform)
    test_dataset = ImageDataset(Config.test_dir, test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4
    )

    # 模型初始化
    model = MADRNet(num_classes=Config.num_classes)
    # model.load_state_dict(torch.load('best_models/DRN_193_0.9629629629629629.pth'), strict=False)
    model = model.to(Config.device)

    criterion = HybridLoss()
    # 优化器分层设置
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': Config.lr_default},  # 主干网络
        {'params': model.rev_blocks.parameters(), 'lr': Config.lr_rev_blocks},  # 可逆块
        {'params': model.embedding_layer.parameters()},  # 嵌入层使用默认lr
        {'params': model.classifier.parameters(), 'lr': Config.lr_classifier}  # 分类头
    ], lr=Config.lr_default, weight_decay=Config.weight_decay)
    num_training_steps = Config.epochs * len(train_loader)

    # OneCycle学习率调度
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        total_steps=num_training_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    logger = MetricLogger(Config.csv_path)
    best_test_accuracy = 0
    # 训练循环
    for epoch in range(1, Config.epochs + 1):

        train_metrics,avg_train_loss = train_epoch(model, train_loader, criterion, optimizer,scheduler, epoch,Config.device,Config.num_classes)

        # 测试阶段
        test_metrics, best_test_accuracy = test_epoch(model, test_loader, criterion, best_test_accuracy, epoch,Config.device,Config.num_classes)

        # 记录指标
        logger.log_metrics(
            epoch=epoch,
            train_metrics=[avg_train_loss] + train_metrics,
            test_metrics=test_metrics
        )

        # 打印epoch摘要
        print(f"\nEpoch {epoch}/{Config.epochs} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_metrics[0]:.4f}")
        print(f"Test Acc: {test_metrics[1]:.4f} | AUC: {test_metrics[5]:.4f}")
        print("----------------------------------")

if __name__ == "__main__":
    main()
