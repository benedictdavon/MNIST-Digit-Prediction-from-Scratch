# MNIST_cnn_predictions.py
import numpy as np
import os, csv, datetime as dt
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from tensorflow import keras

from models.CNN import CNN

import cupy as cp

gpu_count = cp.cuda.runtime.getDeviceCount()
print(f"GPU devices found: {gpu_count}")
if gpu_count == 0:
    raise RuntimeError("No GPU devices found. GPU is required for this script.")

# AFTER (prefer GPU, fallback to CPU)
try:
    USING_GPU = True
except Exception:
    USING_GPU = False

if 'USING_GPU' in globals() and USING_GPU:
    print("[CNN] Using GPU via CuPy")
else:
    print("[CNN] Using CPU (NumPy)")


# ------------------ Helpers ------------------
def _ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _append_row(csv_path: str, header: list, row: list):
    _ensure_parent_dir(csv_path)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)

def _save_matrix_csv(csv_path: str, mat):
    _ensure_parent_dir(csv_path)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        for r in mat:
            w.writerow(list(r))

def _save_history_csv(csv_path: str, history: dict):
    _ensure_parent_dir(csv_path)
    keys = ["epoch", "train_loss", "val_loss", "val_acc"]
    n = max(len(history.get("loss", [])),
            len(history.get("val_loss", [])),
            len(history.get("val_acc", [])))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for i in range(n):
            w.writerow([
                i,
                history.get("loss", [None]*n)[i] if i < len(history.get("loss", [])) else "",
                history.get("val_loss",  [None]*n)[i]  if i < len(history.get("val_loss",  [])) else "",
                history.get("val_acc",   [None]*n)[i]  if i < len(history.get("val_acc",   [])) else "",
            ])


def one_hot(y, num_classes=10):
    y = y.astype(int).ravel()
    oh = np.zeros((num_classes, y.size), dtype=np.float32)
    oh[y, np.arange(y.size)] = 1.0
    return oh.T   # (N, 10)

# Calculate macro precision, recall, and F1-score
def calculate_metrics(cm):
    n_classes = cm.shape[0]
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for i in range(n_classes):
        # True positives for class i
        tp = cm[i, i]
        # False positives for class i (predicted as i but actually other classes)
        fp = np.sum(cm[:, i]) - tp
        # False negatives for class i (actually i but predicted as other classes)
        fn = np.sum(cm[i, :]) - tp
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # F1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    # Macro averages
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    
    return macro_precision, macro_recall, macro_f1, precision_per_class, recall_per_class, f1_per_class

# Plot helpers
def plot_loss(history, tag, filepath):
    os.makedirs(filepath, exist_ok=True)
    plt.figure()
    plt.plot(history.get("loss", []), label="train loss")
    if history.get("val_loss"): plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Loss vs Epochs ({tag})")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{filepath}/loss_curve_{tag}.png", dpi=160); plt.close()

def plot_val_acc(history, tag, filepath):
    if not history.get("val_acc"): return
    os.makedirs(filepath, exist_ok=True)
    plt.figure()
    plt.plot(history["val_acc"], label="val accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy vs Epochs ({tag})")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{filepath}/val_metrics_{tag}.png", dpi=160); plt.close()

def plot_cm(cm, tag, filepath):
    os.makedirs(filepath, exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix ({tag})'); plt.colorbar()
    ticks = np.arange(cm.shape[0]); plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label'); plt.tight_layout()
    plt.savefig(f"{filepath}/confusion_matrix_{tag}.png", dpi=160); plt.close()


# ------------------ Main ------------------
if __name__ == "__main__":
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0,1] but keep 2D structure
    x_train = x_train.astype(np.float32) / 255.0  # (60000, 28, 28)
    x_test = x_test.astype(np.float32) / 255.0    # (10000, 28, 28)
    
    # Add channel dimension for CNN
    X_train = x_train[:, np.newaxis, :, :]  # (60000, 1, 28, 28)
    X_test = x_test[:, np.newaxis, :, :]    # (10000, 1, 28, 28)


    Y_train = one_hot(y_train, 10)
    Y_test  = one_hot(y_test, 10)

    # split off validation set
    val_frac = 0.1
    n_val = int(val_frac * X_train.shape[0])
    X_val, Y_val = X_train[:n_val], Y_train[:n_val]
    X_tr , Y_tr  = X_train[n_val:], Y_train[n_val:]

    # Hyperparameters
    cnn_epochs = 5
    cnn_lr     = 1e-3
    cnn_weight_decay = 1e-4
    cnn_bs     = 128
    cnn_tag    = f"MNIST_CNN_epochs_{cnn_epochs}_lr_{cnn_lr}_bs_{cnn_bs}"

    print(f"Training CNN ({cnn_tag})")
    model = CNN(epochs=cnn_epochs, lr=cnn_lr, batch_size=cnn_bs, seed=42, verbose=1)

    # Train
    history = model.fit(
        X_tr, Y_tr,
        x_val=X_val, y_val=Y_val,
        early_stopping={'monitor': 'val_loss', 'mode': 'min', 'patience': 5},
        tag=cnn_tag, runs_root="runs"
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, Y_test, batch_size=cnn_bs)
    y_pred = model.predict(X_test, batch_size=cnn_bs)

    # 🔍 ADD DEBUG PREDICTIONS HERE
    print("\n" + "="*60)
    print("🚨 DEBUGGING CNN PREDICTIONS")
    print("="*60)
    model.debug_predictions(X_test, y_test, n_samples=20)
    
    # Also debug a few training samples to compare
    print("\n" + "="*60)
    print("🔍 DEBUGGING TRAINING SAMPLES (for comparison)")
    print("="*60)
    model.debug_predictions(X_tr[:20], y_train[n_val:n_val+20], n_samples=20)

    # Confusion matrix
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(y_test, y_pred):
        cm[int(t), int(p)] += 1

    print("Confusion matrix:\n", cm)
    
    macro_precision, macro_recall, macro_f1, precision_per_class, recall_per_class, f1_per_class = calculate_metrics(cm)

    print(f"\nMacro Metrics:")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")

    print(f"\nPer-class metrics:")
    for i in range(10):
        print(f"Class {i}: Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
    # Save outputs
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filepath = f"plots/{cnn_tag}"
    os.makedirs(filepath, exist_ok=True)

    _save_history_csv(os.path.join(filepath, "history.csv"), history)
    _save_matrix_csv(os.path.join(filepath, "confusion_matrix.csv"), cm)

    
    plot_loss(history, cnn_tag, filepath)
    plot_val_acc(history, cnn_tag, filepath)
    plot_cm(cm, cnn_tag, filepath)

    print("\n===== CNN Results =====")
    print("Test acc:", float(test_acc))
    print("Confusion matrix:\n", cm)
