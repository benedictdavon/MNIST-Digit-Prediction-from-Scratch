from ..helpers.Backend import backend

class SGDOptimizer:
    def __init__(self, params, lr=1e-2, weight_decay=0.0):
        self.params = params  # list of [p, g]
        self.lr = lr
        self.wd = weight_decay

    def step(self):
        for p, g in self.params:
            if self.wd != 0.0:
                p -= self.lr * (g + self.wd * p)  # L2 weight decay
            else:
                p -= self.lr * g

    def zero_grad(self):
        for _, g in self.params:
            g[...] = 0.0