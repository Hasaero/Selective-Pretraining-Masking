"""Timer-style DLinear reference: causal full-context next-patch MSE.

Shares Linear((seq_len-patch_len) → patch_len) across all positions, applied
at every token position t ∈ [0, n_pos). Position t's input is the first
(t+1)*patch_len timesteps zero-padded out to seq_len-patch_len; target is
patch t+1. Implemented via einsum + cumsum so all positions are computed
in one shot (no Python loop).

Returns a table of shape (N, global_max_c, n_pos) of per-position MSE.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ._train import set_seed


class DLinearCausalRef(nn.Module):
    """Plain Linear causal next-patch predictor.

    Single Linear(seq_len - patch_len, patch_len) shared across all 7 prediction
    positions via the cumsum trick: weight is reshaped into 7 patch-sized
    sub-blocks, einsum produces per-patch contributions, cumsum builds the
    autoregressive prediction at every position in one forward.
    """

    def __init__(self, seq_len: int, patch_len: int):
        super().__init__()
        self.seq_len   = seq_len
        self.patch_len = patch_len
        self.n_pos     = seq_len // patch_len - 1
        self.linear    = nn.Linear(seq_len - patch_len, patch_len, bias=True)

    def forward(self, x):
        N         = x.shape[0]
        in_len    = self.seq_len - self.patch_len
        patch_len = self.patch_len
        n_pos     = self.n_pos
        W = self.linear.weight                                 # (patch_len, in_len)
        b = self.linear.bias                                   # (patch_len,)
        x_ctx = x[:, :in_len]                                  # (N, in_len)
        x_chunks = x_ctx.reshape(N, n_pos, patch_len)          # (N, P, pl)
        W_chunks = W.reshape(patch_len, n_pos, patch_len)      # (pl_out, P, pl)
        y_chunks = torch.einsum('nki,pki->nkp', x_chunks, W_chunks)
        return y_chunks.cumsum(dim=1) + b                      # (N, n_pos, patch_len)


def build_ref_loss_causal(
    dataset,
    *,
    seq_len: int,
    patch_len: int,
    ref_epochs: int,
    batch_size: int,
    device: torch.device,
    collate_fn,
    seed: int = 42,
    out_dir=None,    # if set, write per-step ref loss to {out_dir}/ref_step_loss.csv
) -> torch.Tensor:
    """Returns ref_table (N, global_max_c, n_pos) on CPU.

    Note: per Timer's behaviour, this uses an *uncapped* ch_mask iteration
    over real channels (filtering c_idx < global_max_c at scatter time)
    so we don't lose coverage during eval.
    """
    set_seed(seed)
    c_sizes      = dataset.channel_counts_per_window()
    global_max_c = int(np.percentile(c_sizes, 99))
    n_pos        = seq_len // patch_len - 1
    # DLinear is tiny (~65k params); use a larger eval batch.
    ref_bs       = max(batch_size * 4, batch_size)

    model     = DLinearCausalRef(seq_len=seq_len, patch_len=patch_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader    = DataLoader(dataset, batch_size=ref_bs, shuffle=True,
                           num_workers=0, collate_fn=collate_fn, pin_memory=False)

    print(f'\n[Ref-Causal] Training DLinear (causal {seq_len}→{patch_len}): '
          f'{ref_epochs} epochs')
    # Per-step ref loss log (if out_dir given)
    step_log_f = None
    if out_dir is not None:
        from pathlib import Path as _P
        _od = _P(out_dir); _od.mkdir(parents=True, exist_ok=True)
        step_log_f = open(_od / 'ref_step_loss.csv', 'w')
        step_log_f.write('step,loss\n')
    global_step = 0
    for epoch in range(ref_epochs):
        model.train()
        total, n = 0.0, 0
        for _, x, ch_mask in tqdm(loader, desc=f'  ref {epoch+1}/{ref_epochs}'):
            x = x.to(device); ch_mask = ch_mask.to(device)
            B, C, T = x.shape
            real_idx = ch_mask.nonzero(as_tuple=False)
            if real_idx.shape[0] == 0:
                continue
            x_real  = x[real_idx[:, 0], real_idx[:, 1]]
            patches = x_real.reshape(-1, T // patch_len, patch_len)
            pred    = model(x_real)
            loss    = ((pred - patches[:, 1:]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            total += loss.item(); n += 1
            global_step += 1
            if step_log_f is not None and global_step % 10 == 0:
                step_log_f.write(f'{global_step},{loss.item():.6f}\n')
                step_log_f.flush()
        print(f'  ref epoch {epoch+1}: mse={total/max(n,1):.4f}')
    if step_log_f is not None:
        step_log_f.close()

    N = len(dataset)
    ref_table = torch.zeros(N, global_max_c, n_pos, dtype=torch.float32)
    model.eval()
    eval_loader = DataLoader(dataset, batch_size=ref_bs, shuffle=False,
                             num_workers=0, collate_fn=collate_fn, pin_memory=False)
    with torch.no_grad():
        for indices, x, ch_mask in tqdm(eval_loader, desc='  ref eval'):
            x = x.to(device); ch_mask = ch_mask.to(device)
            B, C, T = x.shape
            real_idx = ch_mask.nonzero(as_tuple=False)
            if real_idx.shape[0] == 0:
                continue
            keep = real_idx[:, 1] < global_max_c
            real_idx = real_idx[keep]
            if real_idx.shape[0] == 0:
                continue
            b_idx, c_idx = real_idx[:, 0], real_idx[:, 1]
            x_real  = x[b_idx, c_idx]
            patches = x_real.reshape(-1, T // patch_len, patch_len)
            pred    = model(x_real)
            mse_pos = ((pred - patches[:, 1:]) ** 2).mean(dim=-1)
            b_cpu = b_idx.cpu(); c_cpu = c_idx.cpu()
            ref_table[indices[b_cpu], c_cpu] = mse_pos.float().cpu()
    print(f'[Ref-Causal] shape={tuple(ref_table.shape)} '
          f'mean={ref_table.mean():.4f} std={ref_table.std():.4f}')
    return ref_table
