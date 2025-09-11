import numpy as np


class AdamWOptimizer:
    def __init__(
        self, params, lr=1e-3, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8
    ):
        self.params = params  # list of [p, g]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        # state keyed by param id
        self._m = {}
        self._v = {}

    def step(self):
        self.t += 1
        b1t = 1.0 - self.beta1**self.t
        b2t = 1.0 - self.beta2**self.t
        lr = self.lr
        wd = self.weight_decay

        for p, g in self.params:
            pid = id(p)
            if pid not in self._m:
                self._m[pid] = np.zeros_like(p)
                self._v[pid] = np.zeros_like(p)
            m = self._m[pid]
            v = self._v[pid]
            # Adam moments (in-place)
            m[...] = self.beta1 * m + (1.0 - self.beta1) * g
            v[...] = self.beta2 * v + (1.0 - self.beta2) * (g * g)
            m_hat = m / b1t
            v_hat = v / b2t
            # decoupled weight decay
            if wd != 0.0:
                p -= lr * wd * p
            # Adam update
            p -= lr * (m_hat / (np.sqrt(v_hat) + self.eps))
            self._m[pid] = m
            self._v[pid] = v

    def zero_grad(self):
        for _, g in self.params:
            g[...] = 0.0
