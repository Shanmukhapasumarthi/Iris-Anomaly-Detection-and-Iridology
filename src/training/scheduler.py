import math
from typing import Callable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_lambda(warmup_epochs: int,
                          total_epochs:  int) -> Callable[[int], float]:
    """
    Returns a lambda(epoch) → lr_scale factor.
      epoch < warmup_epochs : linear ramp  0 → 1
      epoch >= warmup_epochs: cosine decay 1 → 0
    """
    def _lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return _lambda


def build_scheduler(optimizer:     Optimizer,
                    warmup_epochs: int,
                    total_epochs:  int) -> LambdaLR:
    """Wrap optimizer with warmup + cosine schedule."""
    return LambdaLR(
        optimizer,
        lr_lambda=warmup_cosine_lambda(warmup_epochs, total_epochs)
    )
