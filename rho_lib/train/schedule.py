"""Learning-rate schedule helpers."""
import math

import torch


def warmup_cosine_lambda(num_warmup_steps: int,
                         num_training_steps: int,
                         num_cycles: float = 0.5,
                         min_lr_ratio: float = 0.0):
    """λ(step) for `LambdaLR`, returning a multiplier on the base lr.

    - Linear warmup from 0 → 1 over `num_warmup_steps`
    - Cosine annealing 1 → `min_lr_ratio` over the remaining steps

    `num_cycles=0.5` gives a single half-cosine (default; matches
    transformers/uni2ts cosine_with_restarts at default args).
    """
    def _lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = (float(step - num_warmup_steps) /
                    float(max(1, num_training_steps - num_warmup_steps)))
        if progress >= 1.0:
            return min_lr_ratio
        cosine = 0.5 * (1.0 + math.cos(math.pi * ((num_cycles * progress) % 1.0)))
        return max(min_lr_ratio, cosine)
    return _lambda


def warmup_cosine_with_min_lr_lambda(num_warmup_steps: int,
                                     num_training_steps: int,
                                     warmup_start_lr: float,
                                     base_lr: float,
                                     min_lr: float):
    """MOMENT-style: warmup *starts* at `warmup_start_lr/base_lr`, ends at 1.0;
    then cosine 1 → `min_lr/base_lr`."""
    s_ratio = warmup_start_lr / base_lr
    m_ratio = min_lr / base_lr
    def _lambda(step: int) -> float:
        if step < num_warmup_steps:
            return s_ratio + (1.0 - s_ratio) * step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return m_ratio + 0.5 * (1.0 - m_ratio) * (1.0 + math.cos(math.pi * progress))
    return _lambda


def build_warmup_cosine(optimizer: torch.optim.Optimizer,
                        num_warmup_steps: int,
                        num_training_steps: int,
                        num_cycles: float = 0.5,
                        min_lr_ratio: float = 0.0):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_cosine_lambda(num_warmup_steps, num_training_steps,
                             num_cycles=num_cycles, min_lr_ratio=min_lr_ratio),
    )
