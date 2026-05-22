"""MOMENT-aligned reference: DLinear masked-patch reconstruction.

Trained DLinear that predicts masked patches from the surrounding context
(MAE/BERT style). Used by MOMENT's RHO-CM step in `realtime` mode: the
ref's mask is set equal to MOMENT's per-batch mask so ρ = current − ref
is computed on the SAME (sample, channel, patch) prediction at every
batch — no table mismatch.

API:
- `DLinearMaskedReconRef`            : the trained model (forward applies mask)
- `build_ref_model_masked(...)`      : trains the model on UTSD; returns it
"""
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ._train import set_seed


_TRAIN_MIN_RATIO = 0.15
_TRAIN_MAX_RATIO = 0.5


def _sample_patch_mask(N: int, P: int, ratio: float,
                       device: torch.device) -> torch.Tensor:
    """
    Returns boolean (N, P) mask where True = MASKED (predict target),
    False = observed (input context). Samples ceil(P*ratio) per row.
    """
    n_mask = max(1, int(round(P * ratio)))
    noise  = torch.rand(N, P, device=device)
    idx_sorted = noise.argsort(dim=1)
    masked = torch.zeros(N, P, dtype=torch.bool, device=device)
    masked.scatter_(1, idx_sorted[:, :n_mask], True)
    return masked


class DLinearMaskedReconRef(nn.Module):
    """Plain Linear masked-recon: zero masked positions, run Linear, predict
    full reconstruction. MSE evaluated on masked positions only.
    """

    def __init__(self, seq_len: int, patch_len: int):
        super().__init__()
        self.seq_len   = seq_len
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.linear    = nn.Linear(seq_len, seq_len, bias=True)

    def predict_with_mask(self, x: torch.Tensor,
                          patch_mask: torch.Tensor) -> torch.Tensor:
        """
        x          : (N, T)
        patch_mask : (N, P) bool — True = masked (zero out before linear)
        Returns pred : (N, T) full reconstruction.
        """
        seq_mask = patch_mask.repeat_interleave(self.patch_len, dim=1)
        x_in = torch.where(seq_mask, torch.zeros_like(x), x)
        return self.linear(x_in)


def build_ref_model_masked(
    dataset, *, seq_len, patch_len, ref_epochs, batch_size, device,
    collate_fn, seed: int = 42, lr: float = 1e-3, out_dir=None,
) -> DLinearMaskedReconRef:
    """Train and return a DLinear masked-recon ref. No table is built —
    callers apply MOMENT's per-batch mask to the model at training time."""
    set_seed(seed)
    n_patches = seq_len // patch_len
    model = DLinearMaskedReconRef(seq_len=seq_len, patch_len=patch_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_fn, pin_memory=False)

    print(f'\n[Ref-Masked-Recon] Training masked-recon DLinear: {ref_epochs} epochs')
    rng = random.Random(seed)

    step_log_f = None
    if out_dir is not None:
        from pathlib import Path as _P
        _od = _P(out_dir); _od.mkdir(parents=True, exist_ok=True)
        step_log_f = open(_od / 'ref_step_loss.csv', 'w')
        step_log_f.write('step,loss\n')
    global_step = 0

    for epoch in range(ref_epochs):
        model.train()
        total = 0.0; n = 0
        for _, x, ch_mask in tqdm(loader, desc=f'  ref {epoch+1}/{ref_epochs}'):
            x = x.to(device, non_blocking=True)
            ch_mask = ch_mask.to(device, non_blocking=True)
            B, C, T = x.shape
            x_flat = x.reshape(B * C, T)
            valid  = ch_mask.reshape(B * C).float()
            if not valid.any():
                continue

            ratio = rng.uniform(_TRAIN_MIN_RATIO, _TRAIN_MAX_RATIO)
            patch_mask = _sample_patch_mask(B * C, n_patches, ratio, device)
            seq_mask   = patch_mask.repeat_interleave(patch_len, dim=1)

            pred = model.predict_with_mask(x_flat, patch_mask)
            sq_err = (pred - x_flat) ** 2
            sq_err_masked = sq_err * seq_mask.float()
            denom = seq_mask.float().sum(dim=1).clamp(min=1)
            per_bc = sq_err_masked.sum(dim=1) / denom
            loss = (per_bc * valid).sum() / valid.sum().clamp(min=1)

            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item(); n += 1
            global_step += 1
            if step_log_f is not None and global_step % 10 == 0:
                step_log_f.write(f'{global_step},{loss.item():.6f}\n')
                step_log_f.flush()
        print(f'  ref epoch {epoch+1}: masked_mse={total/max(n,1):.4f}')
    if step_log_f is not None:
        step_log_f.close()
    return model
