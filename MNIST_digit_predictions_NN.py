import numpy as np
from tensorflow import keras
from models.NN.nn_multiclass import NN
from models.NN.nn_multiclass_v2 import NN as NN_v2

from models import CNN

import csv, os, datetime as dt

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
    # Align lengths; if a key is missing, fill with ''
    keys = ["epoch", "train_loss", "val_loss", "val_acc"]
    n = max(len(history.get("train_loss", [])),
            len(history.get("val_loss", [])),
            len(history.get("val_acc", [])))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for i in range(n):
            w.writerow([
                i,
                history.get("train_loss", [None]*n)[i] if i < len(history.get("train_loss", [])) else "",
                history.get("val_loss",  [None]*n)[i]  if i < len(history.get("val_loss",  [])) else "",
                history.get("val_acc",   [None]*n)[i]  if i < len(history.get("val_acc",   [])) else "",
            ])


def load_mnist_keras(val_frac=0.1, seed=0):
    (x_train, y_train), (x_test, y_test) = (
        keras.datasets.mnist.load_data()
    )  # x: (N,28,28), uint8

    # normalize to [0,1] and flatten to (784, m)
    x_train = (
        (x_train.astype(np.float32) / 255.0).reshape(-1, 28 * 28).T
    )  # (784, 60000)
    x_test = (x_test.astype(np.float32) / 255.0).reshape(-1, 28 * 28).T  # (784, 10000)

    # split train → train/val
    rng = np.random.default_rng(seed)
    idx = rng.permutation(x_train.shape[1])
    n_val = int(val_frac * x_train.shape[1])
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_train, Y_train = x_train[:, tr_idx], y_train[tr_idx]
    X_val, Y_val = x_train[:, val_idx], y_train[val_idx]
    X_test, Y_test = x_test, y_test

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def one_hot(y, num_classes=10):
    y = y.astype(int).ravel()
    oh = np.zeros((num_classes, y.size), dtype=np.float32)
    oh[y, np.arange(y.size)] = 1.0
    return oh  # shape: (10, m)


def batch_iter(X, Y, batch_size=128, seed=None):
    m = X.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(m)
    for start in range(0, m, batch_size):
        sel = idx[start : start + batch_size]
        yield X[:, sel], Y[:, sel]


if __name__ == "__main__":
    # Load the MNIST dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist_keras()

    Y_train = one_hot(y_train, 10)  # (10, m_train)
    Y_val = one_hot(y_val, 10)
    Y_test = one_hot(y_test, 10)

    # # ------- Base NN -------
    # # Hyperparameters
    # input_size = X_train.shape[0]
    # hidden_size = 128
    # output_size = 10
    # learning_rate = 0.01
    # epochs = 1000

    # filepath = f"plots/MNIST_epochs_{epochs}_hidden_{hidden_size}_lr_{learning_rate}"

    # model = NN(
    #     n_input=input_size,
    #     n_hidden=hidden_size,
    #     n_output=output_size,
    #     lr=learning_rate,
    #     epochs=epochs,
    # )

    # # full-batch training for quick test
    # history = model.train(X_train, Y_train, x_val=X_val, Y_val=Y_val)

    # # evaluate
    # metrics = model.evaluate(X_test, y_test, Y_onehot=Y_test)
    # print("Test acc:", metrics["accuracy"])
    # print("Macro F1:", metrics["macro_f1"])
    # print("Macro precision:", metrics["macro_precision"])
    # print("Macro recall:", metrics["macro_recall"])
    # print("Confusion matrix:\n", metrics["confusion_matrix"])

    # model.plot_loss(history, tag="mnist", filepath=filepath)
    # model.plot_val_metrics(history, tag="mnist", filepath=filepath)
    # model.plot_confusion_matrix(
    #     metrics["confusion_matrix"], tag="mnist", filepath=filepath
    # )

    # ------- NN with mini-batch -------
    # Hyperparameters
    input_size = X_train.shape[0]

    # Grid search parameters
    # hidden_sizes = [64, 128]
    # learning_rates = [0.001, 0.0005, 0.0001]
    # batch_sizes = [32, 64, 128]
    # epochs = [10, 30, 50, 75]

    hidden_sizes = [128]
    learning_rates = [0.0001]
    batch_sizes = [128]
    epochs = [10, 30, 50, 75]

    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                for epoch in epochs:
                    filepath = f"plots/MNIST_v2_epochs_{epoch}_hidden_{hidden_size}_lr_{learning_rate}_batch_{batch_size}"
                    model_v2 = NN_v2(
                        n_input=input_size,
                        n_hidden=hidden_size,
                        n_output=10,
                        lr=learning_rate,
                        epochs=epoch,
                        batch_size=batch_size,
                    )


                    print(
                        f"Training using NN_v2 with mini-batch size {batch_size} | {epoch} epochs | lr {learning_rate} | batch size {batch_size}"
                    )

                    # mini-batch training
                    history_v2 = model_v2.train(X_train, Y_train, x_val=X_val, Y_val=Y_val)

                    # evaluate
                    metrics_v2 = model_v2.evaluate(X_test, y_test, Y_onehot=Y_test)
                    
                    # ===== SAVE RESULTS =====
                    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Final (last logged) val stats for this run
                    val_loss_last = history_v2["val_loss"][-1] if history_v2.get("val_loss") else ""
                    val_acc_last  = history_v2["val_acc"][-1]  if history_v2.get("val_acc")  else ""

                    # Summary CSV (one row per run)
                    summary_csv = "results/results_summary.csv"
                    summary_header = [
                        "timestamp","model","epochs","hidden_size","lr","batch_size",
                        "val_loss_last","val_acc_last",
                        "test_accuracy","macro_f1","macro_precision","macro_recall"
                    ]
                    summary_row = [
                        timestamp,"NN_v2",epoch,hidden_size,learning_rate,batch_size,
                        val_loss_last,val_acc_last,
                        metrics_v2["accuracy"],metrics_v2["macro_f1"],
                        metrics_v2["macro_precision"],metrics_v2["macro_recall"]
                    ]
                    _append_row(summary_csv, summary_header, summary_row)

                    # Per-class metrics CSV (precision/recall/f1/support)
                    per_class_csv = "results/results_per_class.csv"
                    per_class_header = [
                        "timestamp","model","epochs","hidden_size","lr","batch_size",
                        "class","precision","recall","f1","support"
                    ]
                    prec = metrics_v2["precision"]; rec = metrics_v2["recall"]; f1c = metrics_v2["f1"]; sup = metrics_v2["support"]
                    for cls in range(len(prec)):
                        _append_row(
                            per_class_csv, per_class_header,
                            [timestamp,"NN_v2",epoch,hidden_size,learning_rate,batch_size,
                            cls, float(prec[cls]), float(rec[cls]), float(f1c[cls]), int(sup[cls])]
                        )

                    # Confusion matrix -> CSV (per run)
                    cm_csv_path = os.path.join(filepath, "confusion_matrix_mnist_v2.csv")
                    _save_matrix_csv(cm_csv_path, metrics_v2["confusion_matrix"])

                    # Per-epoch history -> CSV (per run)
                    history_csv_path = os.path.join(filepath, "history.csv")
                    _save_history_csv(history_csv_path, history_v2)
                    # ========================

                    print("Test acc:", metrics_v2["accuracy"])
                    print("Macro F1:", metrics_v2["macro_f1"])
                    print("Macro precision:", metrics_v2["macro_precision"])
                    print("Macro recall:", metrics_v2["macro_recall"])
                    print("Confusion matrix:\n", metrics_v2["confusion_matrix"])
                    model_v2.plot_loss(history_v2, tag="mnist_v2", filepath=filepath)
                    model_v2.plot_val_metrics(history_v2, tag="mnist_v2", filepath=filepath)
                    model_v2.plot_confusion_matrix(
                        metrics_v2["confusion_matrix"], tag="mnist_v2", filepath=filepath
                    )
                    print("--------------------------------------------------\n")

