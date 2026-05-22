"""Per-domain DLinear reference for Moirai.

Each UTSD domain has a single fixed patch_size (DOMAIN_PATCH_SIZES). We
train one Linear ref per domain on its own samples, predicting suffix
patches from lookback patches at the domain-specific patch_size.

Output layout:
    ref_table[domain] : (N_in_domain, C99, n_patches_for_domain)
    Per-patch suffix MSE for each window in the domain.

Lookup at rho time:
    sample_domain = dataset.domain_of(window_idx)
    ref_table[domain][window_local_idx, channel, patch_idx]
"""
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ._train import set_seed


_TRAIN_MIN_RATIO = 0.15
_TRAIN_MAX_RATIO = 0.50
_DEFAULT_REF_SUFFIX_RATIO = 0.30


class DLinearPatchForecastRef(nn.Module):
    """Plain Linear forecaster operating in patch-space.

    Input  : (N, n_patches, patch_size) lookback patches (suffix=0).
    Output : (N, n_patches, patch_size) — predicted patches at every position.
             Loss masks to suffix patches only.
    """

    def __init__(self, n_patches: int, patch_size: int):
        super().__init__()
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.linear = nn.Linear(n_patches * patch_size,
                                n_patches * patch_size,
                                bias=True)

    def forward(self, x_patches):
        N, P, ps = x_patches.shape
        x_flat = x_patches.reshape(N, P * ps)
        out = self.linear(x_flat)
        return out.reshape(N, P, ps)


def _train_one_domain(idxs, dataset, *, seq_len, patch_size,
                      ref_epochs, batch_size, device, collate_fn, seed,
                      domain_label):
    set_seed(seed)
    n_patches = seq_len // patch_size
    if n_patches < 2:
        return None

    sub_dataset = Subset(dataset, idxs)
    loader = DataLoader(sub_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_fn, pin_memory=False)

    model = DLinearPatchForecastRef(n_patches=n_patches,
                                    patch_size=patch_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f'  [{domain_label} ps={patch_size} np={n_patches}] '
          f'{len(sub_dataset)} windows, {len(loader)} batches/epoch')
    rng = random.Random(seed)

    for epoch in range(ref_epochs):
        model.train()
        total = 0.0; n = 0
        desc = f'    ref {epoch+1}/{ref_epochs} {domain_label}'
        for batch in tqdm(loader, desc=desc):
            _, x, ch_mask = batch
            x = x.to(device, non_blocking=True)
            ch_mask = ch_mask.to(device, non_blocking=True)
            B, C, T = x.shape

            ratio = rng.uniform(_TRAIN_MIN_RATIO, _TRAIN_MAX_RATIO)
            n_suffix = max(1, int(round(n_patches * ratio)))
            n_lookback = n_patches - n_suffix
            if n_lookback < 1:
                n_lookback, n_suffix = 1, n_patches - 1

            valid = ch_mask.reshape(B * C)
            if not valid.any():
                continue
            x_flat = x.reshape(B * C, T)

            L_ts = n_lookback * patch_size
            mu = x_flat[:, :L_ts].mean(dim=1, keepdim=True)
            sig_raw = x_flat[:, :L_ts].std(dim=1, keepdim=True)
            nonconst = (sig_raw.squeeze(-1) > 1e-2)
            valid = valid & nonconst
            if not valid.any():
                continue
            sig = sig_raw.clamp(min=1e-2)
            x_n = (x_flat - mu) / sig

            x_patches = x_n.reshape(B * C, n_patches, patch_size)
            x_in = x_patches.clone()
            x_in[:, n_lookback:] = 0.0

            pred = model(x_in)
            sq_err = (pred[:, n_lookback:] - x_patches[:, n_lookback:]) ** 2
            per_patch = sq_err.mean(dim=-1)
            per_bc = per_patch.mean(dim=-1).clamp(max=100.0)

            loss = (per_bc * valid.float()).sum() / valid.float().sum().clamp(min=1)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True); continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item(); n += 1
        print(f'    [{domain_label}] ref epoch {epoch+1}: mse={total/max(n,1):.4f}')

    return model


def build_per_domain_ref_table(
    dataset, *, seq_len, ref_epochs, batch_size, device, collate_fn,
    domain_patch_sizes, default_patch_size,
    seed=42, out_dir=None,
):
    """Train one ref per domain. Returns dict per domain with table."""
    domain_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        domain_to_indices[dataset.domain_of(idx)].append(idx)

    c_sizes = dataset.channel_counts_per_window()
    global_max_c = int(np.percentile(c_sizes, 99))

    print(f'\n[PerDomain-Ref] {len(domain_to_indices)} domains; '
          f'channel_cap={global_max_c}')
    for dom, idxs in domain_to_indices.items():
        ps = domain_patch_sizes.get(dom, (default_patch_size,))[0]
        np_ = seq_len // ps
        print(f'  {dom:<14} {len(idxs):>6} windows  ps={ps}  np={np_}')

    out = {}
    for dom, idxs in domain_to_indices.items():
        ps = domain_patch_sizes.get(dom, (default_patch_size,))[0]
        n_patches = seq_len // ps
        print(f'\n[PerDomain-Ref] Training {dom} (ps={ps})...')
        idxs_list = list(idxs)
        model = _train_one_domain(
            idxs_list, dataset,
            seq_len=seq_len, patch_size=ps,
            ref_epochs=ref_epochs, batch_size=batch_size,
            device=device, collate_fn=collate_fn, seed=seed,
            domain_label=dom,
        )
        if model is None:
            print(f'  [{dom}] SKIP')
            continue

        n_suffix = max(1, int(round(n_patches * _DEFAULT_REF_SUFFIX_RATIO)))
        n_lookback = n_patches - n_suffix
        L_ts = n_lookback * ps

        sub_dataset = Subset(dataset, idxs_list)
        eval_loader = DataLoader(sub_dataset, batch_size=batch_size * 2,
                                 shuffle=False, num_workers=0,
                                 collate_fn=collate_fn, pin_memory=False)
        model.eval()
        G = len(idxs_list)
        table = torch.zeros(G, global_max_c, n_patches, dtype=torch.float32)
        cursor = 0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f'    ref eval {dom}'):
                _, x, ch_mask = batch
                x = x.to(device); B, C, T = x.shape
                C_eff = min(C, global_max_c)
                x_flat = x[:, :C_eff].reshape(B * C_eff, T)
                mu = x_flat[:, :L_ts].mean(dim=1, keepdim=True)
                sig_raw = x_flat[:, :L_ts].std(dim=1, keepdim=True)
                nonconst = (sig_raw.squeeze(-1) > 1e-2).reshape(B, C_eff)
                sig = sig_raw.clamp(min=1e-2)
                x_n = (x_flat - mu) / sig
                x_patches = x_n.reshape(B * C_eff, n_patches, ps)
                x_in = x_patches.clone()
                x_in[:, n_lookback:] = 0.0
                pred = model(x_in)
                sq_err = (pred[:, n_lookback:] - x_patches[:, n_lookback:]) ** 2
                per_patch = sq_err.mean(dim=-1)
                per_patch = per_patch.reshape(B, C_eff, n_suffix)
                per_patch = per_patch * nonconst.float().unsqueeze(-1)
                per_patch_full = torch.zeros(B, C_eff, n_patches,
                                             dtype=torch.float32)
                per_patch_full[:, :, n_lookback:] = per_patch.cpu()
                table[cursor:cursor + B, :C_eff, :] = per_patch_full
                cursor += B

        suffix_only = table[..., n_lookback:]
        print(f'  [{dom}] table shape={tuple(table.shape)} '
              f'suffix_mean={suffix_only.mean():.4f}')

        out[dom] = {
            'patch_size': ps,
            'n_patches': n_patches,
            'window_indices': np.asarray(idxs_list, dtype=np.int64),
            'table': table,
        }

    return out
