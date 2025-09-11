import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import os


'''
Improvements:
- Improving loss and backward to avoid recomputing softmax
- Switch to ReLU + He init (faster learning than tanh)
- Implementing mini-batch gradient descent
- Replacing SGD with Adam
- Add L2 weight decay + Dropout (regularization)
- Implement early stopping based on validation loss 
'''
class NN():
    # Changes: 
    # - He init for ReLU
    # - mini batch training
    # - adding Adam state
    # - add l2 and dropout state
    def __init__(self, n_input, n_hidden, n_output, lr, epochs, batch_size=128, l2_lambda=0.0, dropout_rate=0.0):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Weights initialization (He for ReLU)
        self.W1 = np.random.randn(n_hidden, n_input) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((n_hidden, 1), dtype=np.float32)

        self.W2 = np.random.randn(n_output, n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((n_output, 1), dtype=np.float32)

        # Adam state
        self.t = 0
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)

        # Regularization
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.dropout_mask = None  # to store dropout mask during training

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def relu(self, z): 
        return np.maximum(0.0, z)
    
    def relu_derivative(self, z): 
        return (z > 0).astype(np.float32)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # stability
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    #  Changes: 
    # - compute softmax + CE stably and return the probs so backward can use them directly
    # - add L2 regularization term to loss if l2_lambda > 0
    def loss(self, Y, logits):
        # logits: (C, m)
        m = Y.shape[1]
        z = logits - np.max(logits, axis=0, keepdims=True)   # stabilize
        exp_z = np.exp(z)
        P = exp_z / np.sum(exp_z, axis=0, keepdims=True)
        loss = -np.sum(Y * np.log(P + 1e-12)) / m

        if self.l2_lambda > 0:
            m = Y.shape[1]
            loss += (self.l2_lambda / (2.0*m)) * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return loss, P

    # Changes: 
    # - use ReLU activation
    # - apply dropout during training
    # - add training flag to control dropout
    def forward(self, x, training=False):
        # Hidden layer
        Z1 = self.W1 @ x + self.b1
        A1 = self.relu(Z1)

        # Dropout (training only)
        if training and self.dropout_rate > 0:
            keep = 1.0 - self.dropout_rate
            self.dropout_mask = (np.random.rand(*A1.shape) < keep).astype(np.float32) / keep
            A1 = A1 * self.dropout_mask

        # Output layer (logits)
        Z2 = self.W2 @ A1 + self.b2

        # Cache for backward
        self.Z1, self.A1 = Z1, A1
        self.Z2 = Z2

        return Z2   # return raw logits

    # Changes: 
    # - take P as an argument; don’t use self.P
    # - use ReLU derivative
    # - add L2 regularization term to dW if l2_lambda > 0
    # - apply dropout mask during backprop if used
    def backward(self, X, Y, P=None):
        m = X.shape[1]
        dZ2 = P - Y                        # (10, m)
        dW2 = (1/m) * dZ2 @ self.A1.T
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = self.W2.T @ dZ2

        # Apply dropout mask during backprop if used
        if self.dropout_mask is not None:
            dA1 = dA1 * self.dropout_mask

        dZ1 = dA1 * self.relu_derivative(self.Z1) # ReLu derivative
        dW1 = (1/m) * dZ1 @ X.T
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        if self.l2_lambda > 0:
            m = X.shape[1]
            dW2 += (self.l2_lambda / m) * self.W2
            dW1 += (self.l2_lambda / m) * self.W1
        self.dW1, self.db1, self.dW2, self.db2 = dW1, db1, dW2, db2

    # NEW
    def update_adam(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        lr = self.lr
        # W1
        self.mW1 = beta1*self.mW1 + (1-beta1)*self.dW1
        self.vW1 = beta2*self.vW1 + (1-beta2)*(self.dW1**2)
        mW1_hat = self.mW1 / (1 - beta1**self.t)
        vW1_hat = self.vW1 / (1 - beta2**self.t)
        self.W1 -= lr * mW1_hat / (np.sqrt(vW1_hat) + eps)
        # b1
        self.mb1 = beta1*self.mb1 + (1-beta1)*self.db1
        self.vb1 = beta2*self.vb1 + (1-beta2)*(self.db1**2)
        mb1_hat = self.mb1 / (1 - beta1**self.t)
        vb1_hat = self.vb1 / (1 - beta2**self.t)
        self.b1 -= lr * mb1_hat / (np.sqrt(vb1_hat) + eps)
        # W2
        self.mW2 = beta1*self.mW2 + (1-beta1)*self.dW2
        self.vW2 = beta2*self.vW2 + (1-beta2)*(self.dW2**2)
        mW2_hat = self.mW2 / (1 - beta1**self.t)
        vW2_hat = self.vW2 / (1 - beta2**self.t)
        self.W2 -= lr * mW2_hat / (np.sqrt(vW2_hat) + eps)
        # b2
        self.mb2 = beta1*self.mb2 + (1-beta1)*self.db2
        self.vb2 = beta2*self.vb2 + (1-beta2)*(self.db2**2)
        mb2_hat = self.mb2 / (1 - beta1**self.t)
        vb2_hat = self.vb2 / (1 - beta2**self.t)
        self.b2 -= lr * mb2_hat / (np.sqrt(vb2_hat) + eps)


    # Changes: 
    # - mini-batch training
    # - use Adam update
    # - early stopping based on val loss (if val data provided)
    def train(self, x, Y, x_val=None, Y_val=None, patience=10):
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        m = x.shape[1]

        log_every = max(1, self.epochs // 10)

        best_val = np.inf
        best_snapshot = None
        wait = 0

        for epoch in range(self.epochs):
            # shuffle
            idx = np.random.permutation(m)
            Xs, Ys = x[:, idx], Y[:, idx]

            epoch_loss = 0.0
            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                Xb, Yb = Xs[:, start:end], Ys[:, start:end]

                logits = self.forward(Xb, training=True)  # training=True enables dropout
                loss, P = self.loss(Yb, logits)
                self.backward(Xb, Yb, P)
                self.update_adam()                 # still vanilla for now
                epoch_loss += loss * Xb.shape[1]

            epoch_loss /= m
            history["train_loss"].append(epoch_loss)

            # Validation metrics
            if x_val is not None and Y_val is not None:
                val_logits = self.forward(x_val)          # IMPORTANT: no training-only noise
                val_loss, P_val = self.loss(Y_val, val_logits)
                y_val_int = np.argmax(Y_val, axis=0)
                y_pred_val = np.argmax(P_val, axis=0)
                val_acc = float((y_pred_val == y_val_int).mean())
                history["val_loss"].append(float(val_loss))
                history["val_acc"].append(val_acc)
                
                if epoch % log_every == 0:
                    print(f"Epoch {epoch:3d} | TrainLoss {epoch_loss:.4f} | ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f}")

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_snapshot = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping at epoch {epoch}. Restoring best weights.")
                        self.W1, self.b1, self.W2, self.b2 = best_snapshot
                        break
            else:
                if epoch % log_every == 0:
                    print(f"Epoch {epoch:3d} | TrainLoss {epoch_loss:.4f}")

        if x_val is not None and Y_val is not None:
            print(f"Epoch {self.epochs:3d} | TrainLoss {epoch_loss:.4f} | ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f}")
        else:
            print(f"Epoch {self.epochs:3d} | TrainLoss {epoch_loss:.4f}")

        return history


    def predict(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = np.tanh(Z1)
        Z2 = self.W2 @ A1 + self.b2
        P = self.softmax(Z2)
        return np.argmax(P, axis=0)   # class labels
    
    def evaluate(self, X, y_true, Y_onehot=None, num_classes=10, eps=1e-12):
        # Accept either integer labels (m,) or one-hot (C, m)
        if y_true.ndim == 2:
            # convert one-hot -> integers
            y_true = np.argmax(y_true, axis=0)
        else:
            y_true = y_true.ravel()

        # Forward -> logits and probs
        logits = self.forward(X)
        P = self.softmax(logits)
        y_pred = np.argmax(P, axis=0)

        m = y_true.size

        # Confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1

        # Per-class counts
        TP = np.diag(cm).astype(np.float64)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (TP + FP + FN)

        # Metrics
        precision = TP / (TP + FP + eps)
        recall    = TP / (TP + FN + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        acc       = TP.sum() / m

        macro_precision = precision.mean()
        macro_recall    = recall.mean()
        macro_f1        = f1.mean()

        # Optional loss from P and one-hot
        loss = None
        if Y_onehot is not None:
            true_probs = (P * Y_onehot).sum(axis=0)  # (m,)
            loss = -np.mean(np.log(true_probs + 1e-10))

        metrics = {
            "accuracy": float(acc),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "support": cm.sum(axis=1),
            "confusion_matrix": cm,
            "loss": None if loss is None else float(loss),
        }

        return metrics
    
    def plot_loss(self, history, tag="run", filepath="plots"):
        """Saves loss curve as loss_curve_<tag>.png"""
        os.makedirs(filepath, exist_ok=True)
        plt.figure()
        plt.plot(history["train_loss"], label="train loss")
        plt.plot(history["val_loss"], label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"Loss vs Epochs ({tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{filepath}/loss_curve_{tag}_epochs_{self.epochs}.png", dpi=160)
        plt.close()

    def plot_val_metrics(self, history, tag="run", filepath="plots"):
        """If available, saves val accuracy over epochs as val_metrics_<tag>.png"""
        if "val_acc" not in history or len(history["val_acc"]) == 0:
            return
        os.makedirs(filepath, exist_ok=True)
        plt.figure()
        plt.plot(history["val_acc"], label="val accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Validation Accuracy vs Epochs ({tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{filepath}/val_metrics_{tag}_epochs_{self.epochs}.png", dpi=160)
        plt.close()

    def plot_confusion_matrix(self, cm, tag="run", filepath="plots"):
        """Saves confusion matrix heatmap as confusion_matrix_<tag>.png"""
        os.makedirs(filepath, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=colormap.Blues)
        plt.title(f'Confusion Matrix ({tag})')
        plt.colorbar()
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)

        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(f"{filepath}/confusion_matrix_{tag}.png", dpi=160)
        plt.close()

