import time
import numpy as np

from .layers import (
    Conv2D,
    ReLU,
    MaxPool2D,
    FullyConnectedLayer,
    Flatten,
    Dropout,
)
from .loss.CrossEntropyLoss import CrossEntropyLoss
from .optimizer.SGDOptimizer import SGDOptimizer
from .optimizer.AdamWOptimizer import AdamWOptimizer
from .early_stopping.EarlyStopping import EarlyStopping
from .helpers.logger import RunLogger
from .helpers.Backend import backend

# Remove the xp line since we'll use backend.xp directly when it's available


class CNN:
    def __init__(
        self,
        epochs=10,
        lr=0.01,
        weight_decay=0.0,
        batch_size=128,
        shuffle=True,
        seed=None,
        clip_grad=None,
        verbose=1,
    ):
        # Simple LeNet-5 Style
        self.layers = [
            Conv2D(1, 6, kernel_size=5, stride=1, padding=0),   # (batch, 6, 24, 24)
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),                 # (batch, 6, 12, 12)
            
            Conv2D(6, 16, kernel_size=5, stride=1, padding=0),  # (batch, 16, 8, 8)
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),                 # (batch, 16, 4, 4)
            Flatten(),                                          # (batch, 16*4*4 = 256)
            
            FullyConnectedLayer(16 * 4 * 4, 120),
            ReLU(),
            FullyConnectedLayer(120, 84),
            ReLU(),
            FullyConnectedLayer(84, 10),
        ]

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.verbose = verbose
        self.clip_grad = clip_grad

    def train_mode(self):
        self._training = True

    def eval_mode(self):
        self._training = False

    def forward(self, x, training=False):
        x = backend.ensure_array(x)
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad):
        grad = backend.ensure_array(grad)
        for L in reversed(self.layers):
            grad = L.backward(grad)
        return grad

    def parameters(self):
        ps = []
        for L in self.layers:
            for p, g in zip(L.params(), L.grads()):
                ps.append([p, g])
        return ps

    def predict_proba(self, x, batch_size=256):
        self.eval_mode()
        probs_cpu = []
        for xb in self._batchify(x, batch_size):
            logits = self.forward(xb, training=False)
            
            # Convert to CPU for stable computation
            if backend.use_gpu:
                logits_cpu = backend.to_cpu(logits)
            else:
                logits_cpu = logits
                
            # Use NumPy for softmax calculation (more stable)
            z = logits_cpu - np.max(logits_cpu, axis=1, keepdims=True)
            exp_z = np.exp(z)
            prob = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            probs_cpu.append(prob)
            
        return np.vstack(probs_cpu)

    def predict(self, x, batch_size=256):
        probs = self.predict_proba(x, batch_size=batch_size)
        return np.argmax(probs, axis=1)

    def debug_predictions(self, X_test, y_test, n_samples=10):
        """
        Debug prediction method to analyze model behavior on test samples.

        Args:
            X_test: Test input data
            y_test: True labels (not one-hot encoded)
            n_samples: Number of samples to debug
        """
        print("\n" + "=" * 50)
        print("🔍 DEBUG PREDICTIONS")
        print("=" * 50)

        sample_X = X_test[:n_samples]
        sample_y = y_test[:n_samples]

        print(f"📊 Input Data Analysis:")
        print(f"   Input shape: {sample_X.shape}")
        print(f"   Input min/max: {np.min(sample_X):.4f} / {np.max(sample_X):.4f}")
        print(f"   Input mean/std: {np.mean(sample_X):.4f} / {np.std(sample_X):.4f}")

        # Forward pass with detailed logging
        self.eval_mode()

        # Step 1: Get raw logits
        logits = self.forward(sample_X, training=False)

        # Convert to CPU for analysis
        if backend.use_gpu:
            logits_cpu = backend.to_cpu(logits)
        else:
            logits_cpu = logits

        print(f"\n🧠 Model Forward Pass:")
        print(f"   Logits shape: {logits_cpu.shape}")
        print(f"   Logits min/max: {np.min(logits_cpu):.4f} / {np.max(logits_cpu):.4f}")
        print(
            f"   Logits mean/std: {np.mean(logits_cpu):.4f} / {np.std(logits_cpu):.4f}"
        )

        # Step 2: Apply softmax manually to check
        z = logits_cpu - np.max(logits_cpu, axis=1, keepdims=True)
        exp_z = np.exp(z)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        print(f"\n📈 Softmax Analysis:")
        print(f"   Probabilities shape: {probs.shape}")
        print(
            f"   Probabilities sum per sample: {np.sum(probs, axis=1)[:5]}"
        )  # Should be ~1.0
        print(f"   Max probabilities: {np.max(probs, axis=1)[:5]}")

        # Step 3: Get predictions
        predictions = np.argmax(probs, axis=1)

        print(f"\n🎯 Prediction Results:")
        print(f"   True labels:    {sample_y.tolist()}")
        print(f"   Predictions:    {predictions.tolist()}")
        print(f"   Correct:        {(predictions == sample_y).tolist()}")

        # Step 4: Analyze class distribution in predictions
        unique, counts = np.unique(predictions, return_counts=True)
        print(f"\n📊 Prediction Distribution:")
        for cls, count in zip(unique, counts):
            print(
                f"   Class {cls}: {count}/{n_samples} samples ({count/n_samples*100:.1f}%)"
            )

        # Step 5: Show detailed probabilities for each sample
        print(f"\n🔬 Detailed Sample Analysis:")
        for i in range(min(5, n_samples)):  # Show first 5 samples
            print(f"   Sample {i}: True={sample_y[i]}, Pred={predictions[i]}")
            top3_indices = np.argsort(probs[i])[-3:][::-1]
            print(f"      Top 3 classes: ", end="")
            for j, idx in enumerate(top3_indices):
                print(f"{idx}({probs[i][idx]:.3f})", end="")
                if j < 2:
                    print(", ", end="")
            print()

        # Step 6: Check for systematic bias
        print(f"\n⚠️  Bias Analysis:")
        class_probs_mean = np.mean(probs, axis=0)
        max_bias_class = np.argmax(class_probs_mean)
        print(
            f"   Average probability per class: {[f'{p:.3f}' for p in class_probs_mean]}"
        )
        print(
            f"   Most biased toward class: {max_bias_class} (avg prob: {class_probs_mean[max_bias_class]:.3f})"
        )

        if class_probs_mean[max_bias_class] > 0.5:
            print(f"   ⚠️  WARNING: Strong bias toward class {max_bias_class}!")

        print("=" * 50)

    def evaluate(self, x, y_onehot, loss_fn=None, batch_size=256):
        if loss_fn is None:
            loss_fn = CrossEntropyLoss()

        self.eval_mode()
        N = x.shape[0]
        total_loss = 0.0
        correct = 0
        start = 0
        while start < N:
            end = min(start + batch_size, N)
            xb = backend.ensure_array(x[start:end])
            yb = backend.ensure_array(y_onehot[start:end])
            logits = self.forward(xb, training=False)
            loss, probs = loss_fn.forward(logits, yb)
            total_loss += float(loss) * (end - start)
            probs_cpu = backend.to_cpu(probs) if backend.use_gpu else probs
            yb_cpu = backend.to_cpu(yb) if backend.use_gpu else yb
            pred = np.argmax(probs_cpu, axis=1)
            true = np.argmax(yb_cpu, axis=1)
            correct += np.sum(pred == true)
            start = end
        avg_loss = total_loss / N
        acc = correct / N
        return avg_loss, acc

    def fit(
        self,
        x,
        y_onehot,
        x_val=None,
        y_val=None,
        early_stopping=None,
        patience=5,
        tag="run",
        runs_root="runs",
    ):
        if self.seed is not None:
            self._set_seed(self.seed)

        loss_fn = CrossEntropyLoss()
        optimizer = AdamWOptimizer(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        history = {"loss": [], "acc": []}
        if x_val is not None:
            history["val_loss"] = []
            history["val_acc"] = []

        logger = RunLogger(root=runs_root, tag=tag)
        # normalize early_stopping
        if isinstance(early_stopping, dict):
            stopper = EarlyStopping(**early_stopping)
        else:
            stopper = early_stopping
        # default stopper if none provided but patience given and val set
        if (
            stopper is None
            and x_val is not None
            and y_val is not None
            and patience is not None
        ):
            stopper = EarlyStopping(patience=patience, monitor="val_loss", mode="min")

        print(f"Starting training for {self.epochs} epochs...")
        for ep in range(1, self.epochs + 1):
            t0 = time.time()
            self.train_mode()
            # shuffle
            idx = np.arange(x.shape[0])
            if self.shuffle:
                np.random.shuffle(idx)
            Xs = x[idx]
            Ys = y_onehot[idx]

            # train on shuffled mini-batches (bugfix: use Xs, Ys)
            for xb, yb in self._batchify(Xs, self.batch_size, labels=Ys):
                logits = self.forward(xb, training=True)
                loss, _ = loss_fn.forward(logits, yb)
                grad_logits = loss_fn.backward()
                self.backward(grad_logits)
                if self.clip_grad is not None:
                    self._clip_grad_norm(self.clip_grad)
                optimizer.step()

            # end of epoch: evaluate on train
            train_loss, train_acc = self.evaluate(
                Xs, Ys, batch_size=self.batch_size, loss_fn=CrossEntropyLoss()
            )
            history["loss"].append(train_loss)
            history["acc"].append(train_acc)

            metrics = {"loss": train_loss, "acc": train_acc}

            # validation
            if x_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(
                    x_val, y_val, batch_size=self.batch_size, loss_fn=CrossEntropyLoss()
                )
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                metrics.update({"val_loss": val_loss, "val_acc": val_acc})

            # logging (console)
            if self.verbose > 0:
                log_interval = max(1, self.epochs // 10)
                if ep % log_interval == 0 or ep == 1 or ep == self.epochs:
                    if x_val is not None and y_val is not None:
                        print(
                            f"Epoch {ep}/{self.epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} "
                            f"- val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
                        )
                    else:
                        print(
                            f"Epoch {ep}/{self.epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f}"
                        )

            # logging (files + checkpoints)
            elapsed = time.time() - t0
            if x_val is not None and y_val is not None:
                logger.log_epoch(
                    ep,
                    time_s=elapsed,
                    loss=train_loss,
                    acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                )
                # always save "last"
                logger.save_checkpoint(self._pack_npz_state(), best=False)
                # update best
                if stopper is None:
                    # If no early stopper, still keep best val checkpoint
                    if ep == 1 or val_loss <= np.min(history["val_loss"]):
                        logger.save_checkpoint(self._pack_npz_state(), best=True)
                else:
                    # stopper will restore best weights when it stops; also save a best ckpt
                    if ep == 1 or val_loss <= np.min(history["val_loss"]):
                        logger.save_checkpoint(self._pack_npz_state(), best=True)
            else:
                logger.log_epoch(ep, time_s=elapsed, loss=train_loss, acc=train_acc)
                logger.save_checkpoint(self._pack_npz_state(), best=False)

            # early stopping
            if stopper is not None:
                should_stop = stopper.update(ep, metrics, self)
                if should_stop:
                    if self.verbose > 0:
                        m = stopper.monitor
                        be = stopper.best_epoch
                        bv = stopper.best
                        print(
                            f"Early stopping at epoch {ep:02d}. "
                            f"Best {m}={bv:.4f} at epoch {be:02d}."
                        )
                    # after restore, also save a "best" checkpoint reflecting restored params
                    logger.save_checkpoint(self._pack_npz_state(), best=True)
                    break

        # finalize history file
        logger.save_json()
        return history

    # ================== helpers ==================
    def _batchify(self, X, batch_size, labels=None):
        N = X.shape[0]
        start = 0
        if labels is None:
            while start < N:
                end = min(start + batch_size, N)
                xb = X[start:end]
                yield backend.ensure_array(xb)
                start = end
        else:
            while start < N:
                end = min(start + batch_size, N)
                xb = X[start:end]
                yb = labels[start:end]
                yield backend.ensure_array(xb), backend.ensure_array(yb)
                start = end

    def _clip_grad_norm(self, max_norm):
        """
        Clips all parameter grads to have global L2 norm <= max_norm.
        """
        normsqr = 0.0
        for _, g in self.parameters():
            normsqr += np.sum(g * g)
        total_norm = np.sqrt(normsqr) + 1e-12
        if total_norm > max_norm:
            scale = max_norm / total_norm
            for _, g in self.parameters():
                g *= scale

    def _snapshot_params(self):
        # Deep-copy all params into a flat list (for early stopping).
        snap = []
        for p, _ in self.parameters():
            snap.append(np.copy(p))
        return snap

    def _load_params(self, snapshot):
        # Load params from a flat list created by _snapshot_params().
        i = 0
        for p, _ in self.parameters():
            p[...] = snapshot[i]
            i += 1

    def _pack_npz_state(self):
        # Pack params into a dict for RunLogger checkpoint
        arrays = {}
        i = 0
        for L in self.layers:
            for p, _ in zip(L.params(), L.grads()):
                arrays[f"p{i}"] = p
                i += 1
        return arrays

    # model I/O
    def save(self, path):
        arrays = {}
        i = 0
        for L in self.layers:
            for p, _ in zip(L.params(), L.grads()):
                arrays[f"p{i}"] = p
                i += 1
        np.savez(path, **arrays)

    def load(self, path):
        data = np.load(path)
        i = 0
        for L in self.layers:
            for p, _ in zip(L.params(), L.grads()):
                p[...] = data[f"p{i}"]
                i += 1

    def _set_seed(self, seed=42):
        np.random.seed(seed)
