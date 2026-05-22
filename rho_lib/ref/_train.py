"""Shared DLinear training boilerplate."""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def train_dlinear(
    model: torch.nn.Module,
    dataset,
    *,
    ref_epochs: int,
    batch_size: int,
    device: torch.device,
    collate_fn,
    step_fn,        # called as step_fn(model, x, ch_mask) -> scalar loss
    tag: str = 'Ref',
):
    """Train `model` on `dataset` for `ref_epochs` epochs.
    `step_fn` does the per-batch forward + loss; we handle the optimizer."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_fn, pin_memory=False)
    print(f'\n[{tag}] Training DLinear: {ref_epochs} epochs')
    for epoch in range(ref_epochs):
        model.train()
        total, n = 0.0, 0
        for _, x, ch_mask in tqdm(loader, desc=f'  ref {epoch+1}/{ref_epochs}'):
            x = x.to(device, non_blocking=True)
            ch_mask = ch_mask.to(device, non_blocking=True)
            loss = step_fn(model, x, ch_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item(); n += 1
        print(f'  ref epoch {epoch+1}: loss={total/max(n,1):.4f}')
    return model
