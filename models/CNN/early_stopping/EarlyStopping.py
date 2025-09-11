
class EarlyStopping:
    def __init__(
        self,
        patience=5,
        min_delta=0.0,
        monitor="val_loss",
        mode="min",
        restore_best_weights=True,
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = float(min_delta)
        self.restore_best_weights = restore_best_weights

        self.best = None
        self.best_epoch = -1
        self.wait = 0
        self.stopped = False
        self._best_snapshot = None

    def _is_better(self, current, best):
        if self.mode == "min":
            return current < (best - self.min_delta)
        else:  # 'max'
            return current > (best + self.min_delta)

    def update(self, epoch, metrics, model):
        value = metrics[self.monitor]
        if self.best is None or self._is_better(value, self.best):
            self.best = value
            self.best_epoch = epoch
            self.wait = 0
            # snapshot best weights
            self._best_snapshot = model._snapshot_params()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True
                if self.restore_best_weights and self._best_snapshot is not None:
                    model._load_params(self._best_snapshot)
                return True
        return False

