import torch
import numpy as np
from .calculate_metrics import calculate_metrics

def test_epoch(model, loader, criterion, best_test_accuracy, epoch,device,num_classes):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets, ratios in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            ratios = ratios.to(device)

            outputs, embeddings = model(inputs)
            loss = criterion(outputs, targets, embeddings, ratios)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy())
            total_loss += loss.item()

    metrics = calculate_metrics(all_targets, all_preds, all_probs,num_classes)
    avg_loss = total_loss / len(loader)

    # if  epoch == 200:
    #     model_path = f'result/who_fold_1_70_last_model_epoch_{best_test_accuracy}.pth'
    #     torch.save(model.state_dict(), model_path)
    #     print(f'-> 保存最后的模型: {model_path} ')

    if metrics[0] > best_test_accuracy:

        best_test_accuracy = metrics[0]
        # 保存最佳模型
        if best_test_accuracy >= 0.90:
            model_path = f'result/who_fold_1_ENH_best_{best_test_accuracy}.pth'
            torch.save(model.state_dict(), model_path)
            print(f'-> 保存最佳模型: {model_path} ')
    return [avg_loss] + metrics, best_test_accuracy
