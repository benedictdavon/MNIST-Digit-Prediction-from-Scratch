import numpy as np
from ..helpers.Backend import backend

class CrossEntropyLoss:
    def __init__(self, eps=1e-12):
        self.eps = eps
        # cache from forward
        self.probs = None
        self.m = None
        self.Y = None

    def forward(self, logits, Y_onehot):
        """
        logits: (batch, num_classes)  -- pre-softmax
        Y_onehot: (batch, num_classes)
        returns: (loss_scalar, probs)
        """
        logits = backend.ensure_array(logits)
        Y_onehot = backend.ensure_array(Y_onehot)
        
        self.m = Y_onehot.shape[0]
        self.Y = Y_onehot

        # Stable log-softmax
        z = logits - backend.max(logits, axis=1, keepdims=True)  # (B, C)
        logsumexp = backend.log(backend.sum(backend.exp(z), axis=1, keepdims=True))  # (B, 1)
        log_probs = z - logsumexp  # (B, C)

        # Cross-entropy: -sum(Y * log_probs) / B
        loss = -backend.sum(Y_onehot * log_probs) / self.m

        # Also cache probs for backward (and for any caller needing them)
        probs = backend.exp(log_probs)
        self.probs = probs

        # Convert loss to scalar if needed
        if backend.use_gpu:
            loss = backend.to_cpu(loss).item()
        else:
            loss = float(loss)

        return loss, probs

    def backward(self):
        """
        dL/dlogits = (probs - Y)/m
        This is the fused softmax+CE gradient.
        """
        if self.probs is None or self.Y is None or self.m is None:
            raise ValueError("Must call forward() before backward()")
        return (self.probs - self.Y) / self.m

