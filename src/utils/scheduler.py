"""WSD (Warmup-Stable-Decay) learning rate scheduler.

Reference: MiniCPM (Hu et al., 2024, arXiv:2404.06395)
"""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def create_wsd_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.01,
    decay_ratio: float = 0.10,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create a WSD (Warmup-Stable-Decay) learning rate scheduler.

    Three phases:
        1. Warmup: linear increase from 0 to peak LR
        2. Stable: constant at peak LR
        3. Decay: cosine annealing from peak LR to min LR

    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of training steps
        warmup_ratio: Fraction of total steps for warmup phase
        decay_ratio: Fraction of total steps for decay phase
        min_lr_ratio: Minimum LR as fraction of peak LR (0.0 = decay to zero)

    Returns:
        LambdaLR scheduler
    """
    warmup_steps = int(total_steps * warmup_ratio)
    decay_steps = int(total_steps * decay_ratio)
    stable_steps = total_steps - warmup_steps - decay_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return current_step / max(1, warmup_steps)
        elif current_step < warmup_steps + stable_steps:
            # Stable phase
            return 1.0
        else:
            # Cosine decay
            progress = (current_step - warmup_steps - stable_steps) / max(1, decay_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
