import csv


class MetricLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_acc', 'train_precision', 'train_recall',
                'train_f1', 'train_auc', 'test_loss', 'test_acc', 'test_precision', 'test_recall',
                'test_f1', 'test_auc'
            ])

    def log_metrics(self, epoch, train_metrics, test_metrics):
        row = [epoch] + train_metrics + test_metrics
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)