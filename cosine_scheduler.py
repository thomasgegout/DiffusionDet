"""
Enhanced LR Scheduler for DiffusionDet
Adds Cosine Annealing LR scheduler for better convergence
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from detectron2.solver.build import LR_SCHEDULER_REGISTRY


@LR_SCHEDULER_REGISTRY.register()
class CosineAnnealingWarmupLR(_LRScheduler):
    """
    Cosine Annealing LR Scheduler with Warmup
    
    Args:
        optimizer: Wrapped optimizer
        max_iter: Maximum training iterations
        warmup_iters: Number of warmup iterations
        warmup_factor: Initial warmup learning rate factor
        eta_min_ratio: Minimum LR ratio at the end (default: 0.01)
    """
    
    def __init__(self, optimizer, max_iter, warmup_iters=1000, warmup_factor=0.001, eta_min_ratio=0.01, last_epoch=-1):
        self.max_iter = max_iter
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.eta_min_ratio = eta_min_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Warmup phase: linear increase
            alpha = self.last_epoch / self.warmup_iters
            warmup_lr = [
                base_lr * (self.warmup_factor * (1 - alpha) + alpha)
                for base_lr in self.base_lrs
            ]
            return warmup_lr
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_iters) / (self.max_iter - self.warmup_iters)
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            
            cosine_lr = [
                self.eta_min_ratio * base_lr + 
                (base_lr - self.eta_min_ratio * base_lr) * 
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]
            return cosine_lr


def build_cosine_annealing_lr_scheduler(cfg, optimizer):
    """
    Build cosine annealing LR scheduler
    """
    return CosineAnnealingWarmupLR(
        optimizer,
        max_iter=cfg.SOLVER.MAX_ITER,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        eta_min_ratio=0.01  # End with 1% of base LR
    )
