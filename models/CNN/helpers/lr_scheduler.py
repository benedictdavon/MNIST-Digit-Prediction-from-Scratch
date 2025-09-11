"""
Learning Rate Schedulers for neural network training.
Supports various scheduling strategies including step decay, exponential decay, cosine annealing, etc.
"""
import numpy as np
import math


class LRScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, optimizer, verbose=True):
        self.optimizer = optimizer
        self.verbose = verbose
        self.initial_lr = optimizer.lr
        self.current_lr = optimizer.lr
        
    def step(self, epoch, metrics=None):
        """Update learning rate based on epoch and optionally metrics."""
        new_lr = self.get_lr(epoch, metrics)
        if new_lr != self.current_lr:
            self.current_lr = new_lr
            self.optimizer.lr = new_lr
            if self.verbose:
                print(f"   📉 LR updated: {new_lr:.6f}")
        return new_lr
    
    def get_lr(self, epoch, metrics=None):
        """Override this method in subclasses."""
        return self.current_lr


class StepLR(LRScheduler):
    """Step decay: reduce LR by gamma every step_size epochs."""
    
    def __init__(self, optimizer, step_size, gamma=0.1, verbose=True):
        super().__init__(optimizer, verbose)
        self.step_size = step_size
        self.gamma = gamma
        
    def get_lr(self, epoch, metrics=None):
        return self.initial_lr * (self.gamma ** (epoch // self.step_size))


class ExponentialLR(LRScheduler):
    """Exponential decay: LR = initial_lr * gamma^epoch."""
    
    def __init__(self, optimizer, gamma=0.95, verbose=True):
        super().__init__(optimizer, verbose)
        self.gamma = gamma
        
    def get_lr(self, epoch, metrics=None):
        return self.initial_lr * (self.gamma ** epoch)


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing: smooth cosine decay from initial_lr to min_lr."""
    
    def __init__(self, optimizer, T_max, min_lr=0, verbose=True):
        super().__init__(optimizer, verbose)
        self.T_max = T_max
        self.min_lr = min_lr
        
    def get_lr(self, epoch, metrics=None):
        return self.min_lr + (self.initial_lr - self.min_lr) * \
               (1 + math.cos(math.pi * epoch / self.T_max)) / 2


class ReduceLROnPlateau(LRScheduler):
    """Reduce LR when metric stops improving (like validation loss)."""
    
    def __init__(self, optimizer, monitor='val_loss', mode='min', factor=0.5, 
                 patience=10, threshold=1e-4, min_lr=0, verbose=True):
        super().__init__(optimizer, verbose)
        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = {'min': np.inf, 'max': -np.inf}[mode]
        
    def get_lr(self, epoch, metrics=None):
        if metrics is None or self.monitor not in metrics:
            return self.current_lr
            
        current = metrics[self.monitor]
        
        if self.best is None:
            self.best = current
        else:
            if self.mode == 'min':
                improved = current < self.best - self.threshold
            else:  # mode == 'max'
                improved = current > self.best + self.threshold
                
            if improved:
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
                
        if self.num_bad_epochs >= self.patience:
            new_lr = max(self.current_lr * self.factor, self.min_lr)
            if new_lr < self.current_lr:
                self.num_bad_epochs = 0
                if self.verbose:
                    print(f"   📉 ReduceLROnPlateau: {self.monitor} didn't improve for {self.patience} epochs")
                return new_lr
                
        return self.current_lr


class WarmupCosineScheduler(LRScheduler):
    """Warmup followed by cosine annealing decay."""
    
    def __init__(self, optimizer, warmup_epochs=5, max_epochs=100, 
                 warmup_lr=1e-6, min_lr=1e-6, verbose=True):
        super().__init__(optimizer, verbose)
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        
    def get_lr(self, epoch, metrics=None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.warmup_lr + (self.initial_lr - self.warmup_lr) * epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return self.min_lr + (self.initial_lr - self.min_lr) * \
                   (1 + math.cos(math.pi * progress)) / 2


class MultiStepLR(LRScheduler):
    """Decay LR by gamma at specified milestones."""
    
    def __init__(self, optimizer, milestones, gamma=0.1, verbose=True):
        super().__init__(optimizer, verbose)
        self.milestones = sorted(milestones)
        self.gamma = gamma
        
    def get_lr(self, epoch, metrics=None):
        gamma_power = sum(1 for milestone in self.milestones if epoch >= milestone)
        return self.initial_lr * (self.gamma ** gamma_power)


class CyclicLR(LRScheduler):
    """Cyclic learning rate with triangular policy."""
    
    def __init__(self, optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=2000, 
                 mode='triangular', verbose=True):
        super().__init__(optimizer, verbose)
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.mode = mode
        self.step_count = 0
        
    def get_lr(self, epoch, metrics=None):
        cycle = np.floor(1 + self.step_count / (2 * self.step_size_up))
        x = np.abs(self.step_count / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        else:  # Can add more modes
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
            
        self.step_count += 1
        return lr


def get_scheduler(name, optimizer, **kwargs):
    """Factory function to create schedulers by name."""
    schedulers = {
        'step': StepLR,
        'exponential': ExponentialLR,
        'cosine': CosineAnnealingLR,
        'plateau': ReduceLROnPlateau,
        'warmup_cosine': WarmupCosineScheduler,
        'multistep': MultiStepLR,
        'cyclic': CyclicLR,
    }
    
    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(schedulers.keys())}")
        
    return schedulers[name](optimizer, **kwargs)
