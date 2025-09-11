import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import os

class NN():
    def __init__(self, n_input, n_hidden, n_output, lr, epochs):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # xavier initialization
        self.W1 = np.random.randn(n_hidden, n_input) * np.sqrt(1 / n_input)
        # small constant
        self.b1 = np.full((n_hidden, 1), 0.01)

        self.W2 = np.random.randn(n_output, n_hidden) * np.sqrt(1 / n_hidden)
        self.b2 = np.full((n_output, 1), 0.01)

        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # stability
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def loss(self, Y, Z2):
        m = Y.shape[1]
        P = self.softmax(Z2)
        loss = -np.sum(Y * np.log(P + 1e-10)) / m
        self.P = P  # cache softmax probs
        return loss

    def forward(self, x):
        # Hidden layer
        Z1 = self.W1 @ x + self.b1
        A1 = np.tanh(Z1)

        # Output layer (logits)
        Z2 = self.W2 @ A1 + self.b2

        # Cache for backward
        self.Z1, self.A1 = Z1, A1
        self.Z2 = Z2

        return Z2   # return raw logits

    def backward(self, X, Y):
        m = X.shape[1]
        dZ2 = self.P - Y                        # (10, m)
        dW2 = (1/m) * dZ2 @ self.A1.T
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * (1 - self.A1**2)           # tanh derivative
        dW1 = (1/m) * dZ1 @ X.T
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        self.dW1, self.db1, self.dW2, self.db2 = dW1, db1, dW2, db2

    def update(self):
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

    def train(self, x, Y, x_val=None, Y_val=None):
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        for epoch in range(self.epochs):
            A2 = self.forward(x)

            loss = self.loss(Y, A2)
            history["train_loss"].append(loss)

            self.backward(x, Y)

            self.update()

            # Validation metrics
            if x_val is not None and Y_val is not None:
                val_logits = self.forward(x_val)
                val_loss = self.loss(Y_val, val_logits)
                history["val_loss"].append(val_loss)
                
                val_metrics = self.evaluate(x_val, Y_val, Y_val)
                val_acc = val_metrics["accuracy"]
                history["val_acc"].append(val_acc)

            log_every = max(1, self.epochs // 10)
            if epoch % log_every == 0:
                if x_val is not None and Y_val is not None:
                    print(f"Epoch {epoch}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
        if x_val is not None and Y_val is not None:
            print(f"Epoch {self.epochs}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {self.epochs}, Loss: {loss:.4f}")
        
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

