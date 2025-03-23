import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(y_true, y_pred, y_prob,num_classes):
    acc = accuracy_score(y_true, y_pred)

    # 设置 zero_division=0 避免警告
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 多分类AUC计算（需要one-hot编码）
    y_true_onehot = np.eye(num_classes)[y_true]
    try:
        auc = roc_auc_score(y_true_onehot, y_prob, multi_class='ovr')
    except ValueError:
        auc = 0.5  # 处理类别样本不均衡情况

    return [acc, precision, recall, f1, auc]