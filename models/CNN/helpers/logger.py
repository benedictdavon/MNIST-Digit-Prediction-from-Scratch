# helpers/logger.py
import numpy as np
import csv, json, time, datetime, pathlib, shutil
import os
import matplotlib.pyplot as plt
import matplotlib.cm as colormap


class RunLogger:
    def __init__(self, root="runs", tag="run"):
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.root = pathlib.Path(root)
        self.dir = self.root / f"{tag}_{ts}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "history.csv"
        self.json_path = self.dir / "history.json"
        self.best_ckpt = self.dir / "checkpoint_best.npz"
        self.last_ckpt = self.dir / "checkpoint_last.npz"
        self.metrics = []  # list of dicts per epoch
        self._csv_header_written = False

    # ---------- logging ----------
    def log_epoch(self, epoch, **kwargs):
        row = {"epoch": int(epoch), **{k: float(v) for k, v in kwargs.items()}}
        self.metrics.append(row)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)

    def save_json(self):
        with open(self.json_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def save_checkpoint(self, npz_dict, best=False):
        path = self.best_ckpt if best else self.last_ckpt
        np.savez(path, **npz_dict)
        return str(path)

    def copy_plots_from(self, plots_dir):
        plots_dir = pathlib.Path(plots_dir)
        if plots_dir.exists():
            for p in plots_dir.glob("*.png"):
                shutil.copy2(p, self.dir / p.name)

    # ---------- plotting (new) ----------
    def _plots_dir(self, subdir):
        out = self.dir / subdir
        out.mkdir(parents=True, exist_ok=True)
        return out

    def plot_loss(self, history, tag="run", subdir="plots", total_epochs=None):
        """
        Saves loss curve as loss_curve_<tag>.png.
        Accepts history with either keys:
          - {'loss': [...], 'val_loss': [...]}
          - or {'train_loss': [...], 'val_loss': [...]}
        """
        train_key = "loss" if "loss" in history else "train_loss"
        train = history.get(train_key, [])
        val = history.get("val_loss", [])

        outdir = self._plots_dir(subdir)
        plt.figure()
        if len(train) > 0:
            plt.plot(train, label="train loss")
        if len(val) > 0:
            plt.plot(val, label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        if total_epochs is None:
            total_epochs = max(len(train), len(val))
        plt.title(f"Loss vs Epochs ({tag})")
        if len(train) > 0 or len(val) > 0:
            plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"loss_curve_{tag}_epochs_{total_epochs}.png", dpi=160)
        plt.close()

    def plot_val_metrics(self, history, tag="run", subdir="plots", total_epochs=None):
        """
        Saves validation accuracy curve as val_metrics_<tag>.png if 'val_acc' is present.
        """
        val_acc = history.get("val_acc", [])
        if len(val_acc) == 0:
            return
        outdir = self._plots_dir(subdir)
        plt.figure()
        plt.plot(val_acc, label="val accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        if total_epochs is None:
            total_epochs = len(val_acc)
        plt.title(f"Validation Accuracy vs Epochs ({tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"val_metrics_{tag}_epochs_{total_epochs}.png", dpi=160)
        plt.close()

    def plot_confusion_matrix(self, cm, tag="run", subdir="plots", class_names=None):
        """
        Saves confusion matrix heatmap as confusion_matrix_<tag>.png
        cm: (num_classes, num_classes) integer matrix
        """
        outdir = self._plots_dir(subdir)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=colormap.Blues)
        plt.title(f"Confusion Matrix ({tag})")
        plt.colorbar()
        n_classes = cm.shape[0]
        ticks = np.arange(n_classes)
        if class_names is None:
            class_names = ticks
        plt.xticks(ticks, class_names, rotation=0)
        plt.yticks(ticks, class_names)

        thresh = cm.max() / 2.0 if cm.size > 0 else 0
        for i, j in np.ndindex(cm.shape):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(outdir / f"confusion_matrix_{tag}.png", dpi=160)
        plt.close()

    def plot_all(self, history, tag="run", subdir="plots"):
        """
        Convenience: generate all standard plots we know how to draw.
        """
        total_epochs = max(
            len(history.get("loss", [])),
            len(history.get("train_loss", [])),
            len(history.get("val_loss", [])),
        )
        self.plot_loss(history, tag=tag, subdir=subdir, total_epochs=total_epochs)
        self.plot_val_metrics(history, tag=tag, subdir=subdir, total_epochs=total_epochs)

    # ---------- metrics calculation ----------
    def calculate_metrics_from_confusion_matrix(self, cm, eps=1e-12):
        """
        Calculate macro F1, precision, and recall from confusion matrix.
        
        Args:
            cm: (n_classes, n_classes) confusion matrix where cm[i,j] = true_label_i predicted as j
            eps: small value to avoid division by zero
            
        Returns:
            dict with 'macro_precision', 'macro_recall', 'macro_f1', 'accuracy'
        """
        n_classes = cm.shape[0]
        
        # Per-class metrics
        precision_per_class = np.zeros(n_classes)
        recall_per_class = np.zeros(n_classes)
        f1_per_class = np.zeros(n_classes)
        
        for i in range(n_classes):
            # True positives for class i
            tp = cm[i, i]
            
            # False positives for class i (predicted as i but actually other classes)
            fp = np.sum(cm[:, i]) - tp
            
            # False negatives for class i (actually i but predicted as other classes)
            fn = np.sum(cm[i, :]) - tp
            
            # Precision: TP / (TP + FP)
            precision_per_class[i] = tp / (tp + fp + eps)
            
            # Recall: TP / (TP + FN)  
            recall_per_class[i] = tp / (tp + fn + eps)
            
            # F1: 2 * (precision * recall) / (precision + recall)
            p, r = precision_per_class[i], recall_per_class[i]
            f1_per_class[i] = 2 * p * r / (p + r + eps)
        
        # Macro averages (unweighted mean across classes)
        macro_precision = np.mean(precision_per_class)
        macro_recall = np.mean(recall_per_class)
        macro_f1 = np.mean(f1_per_class)
        
        # Overall accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        
        return {
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall), 
            'macro_f1': float(macro_f1),
            'accuracy': float(accuracy),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist()
        }
    
    def calculate_metrics_from_predictions(self, y_true, y_pred, num_classes=10, eps=1e-12):
        """
        Calculate metrics directly from true and predicted labels.
        
        Args:
            y_true: array of true class labels (integers)
            y_pred: array of predicted class labels (integers)
            num_classes: number of classes
            eps: small value to avoid division by zero
            
        Returns:
            dict with metrics and confusion matrix
        """
        # Build confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for true_label, pred_label in zip(y_true, y_pred):
            cm[int(true_label), int(pred_label)] += 1
        
        # Calculate metrics from confusion matrix
        metrics = self.calculate_metrics_from_confusion_matrix(cm, eps=eps)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def save_metrics_summary(self, metrics, tag="run", filename="metrics_summary.json"):
        """
        Save detailed metrics summary to JSON file.
        
        Args:
            metrics: dict containing calculated metrics
            tag: experiment tag
            filename: output filename
        """
        summary = {
            'experiment_tag': tag,
            'timestamp': datetime.datetime.now().isoformat(),
            'overall_metrics': {
                'accuracy': metrics.get('accuracy', 0.0),
                'macro_precision': metrics.get('macro_precision', 0.0),
                'macro_recall': metrics.get('macro_recall', 0.0),
                'macro_f1': metrics.get('macro_f1', 0.0)
            },
            'per_class_metrics': {
                'precision': metrics.get('precision_per_class', []),
                'recall': metrics.get('recall_per_class', []),
                'f1': metrics.get('f1_per_class', [])
            }
        }
        
        # Add confusion matrix if present
        if 'confusion_matrix' in metrics:
            summary['confusion_matrix'] = metrics['confusion_matrix'].tolist()
        
        output_path = self.dir / filename
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(output_path)
