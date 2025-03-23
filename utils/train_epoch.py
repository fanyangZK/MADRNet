import  torch
import numpy as np
from .calculate_metrics import calculate_metrics
def train_epoch(model, train_loader, criterion,optimizer,scheduler, epoch,device,num_classes):
    model.train()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_targets = []

    for batch_idx, (inputs, targets, ratios) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        ratios = ratios.to(device)

        optimizer.zero_grad()
        outputs, embeddings = model(inputs)  # 获取嵌入向量
        loss = criterion(outputs, targets, embeddings, ratios)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # 每个step更新学习率

        # 收集训练指标
        probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_targets.extend(targets.cpu().numpy())
        total_loss += loss.item()

        # 每100个batch打印进度
        if batch_idx % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} | Batch {batch_idx} | LR: {current_lr:.2e} | Loss: {loss.item():.4f}")

    # 计算epoch指标
    train_metrics = calculate_metrics(all_targets, all_preds, all_probs,num_classes)
    avg_train_loss = total_loss / len(train_loader)
    return  train_metrics,avg_train_loss