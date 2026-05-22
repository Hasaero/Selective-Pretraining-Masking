"""Timer few-shot linear probing.

Loads pretrained Timer encoder, freezes it, and trains a small linear
forecasting head from last-patch hidden state to horizon. Subsamples
training windows by --train-frac.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Reuse model + data prep from pretrain_timer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pretrain_timer import (
    TimerModel, prepare_forecast_datasets,
    DATA_DIR, SEQ_LEN, PATCH_LEN,
)


class TimerLinearProbe(nn.Module):
    """Frozen Timer encoder + linear forecasting head.

    Channel-independent: input (B, C, sl) -> reshape to (B*C, n_patches, patch_len).
    RevIN per (sample, channel) using stats from input window only.
    Head: last-patch hidden state (d_model) -> horizon.
    """
    def __init__(self, encoder: TimerModel, horizon: int):
        super().__init__()
        self.encoder = encoder
        self.horizon = horizon
        self.head = nn.Linear(encoder.d_model, horizon, bias=True)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def encode(self, x_patches):
        # x_patches: (N, n_patches, patch_len), already RevIN-normalised
        h = self.encoder.patch_embed(x_patches)
        for block in self.encoder.blocks:
            h = block(h)
        h = self.encoder.norm(h)
        return h  # (N, n_patches, d_model)

    def forward(self, x_bcs):
        # x_bcs: (B, C, sl)
        B, C, sl = x_bcs.shape
        assert sl % PATCH_LEN == 0, f'sl={sl} not divisible by patch_len={PATCH_LEN}'
        n_patches = sl // PATCH_LEN

        x_flat = x_bcs.reshape(B * C, sl)
        mu  = x_flat.mean(dim=-1, keepdim=True)
        sig = x_flat.std(dim=-1, keepdim=True) + 1e-5
        x_norm = (x_flat - mu) / sig
        x_patches = x_norm.reshape(B * C, n_patches, PATCH_LEN)

        with torch.no_grad():
            h = self.encode(x_patches)         # (N, n_patches, d_model)
        last = h[:, -1, :]                     # (N, d_model)
        y = self.head(last)                    # (N, horizon)
        y = y * sig + mu                       # de-norm
        y = y.reshape(B, C, self.horizon)
        return y


def linear_probe_one(ckpt_path, dataset_name, horizon, *,
                     seed, device, train_frac=1.0,
                     epochs=10, batch_size=256, lr=1e-3,
                     context_length=SEQ_LEN, num_workers=0, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    train_ds, val_ds, test_ds = prepare_forecast_datasets(
        dataset_name, DATA_DIR, context_length, horizon, stride=1
    )

    if train_frac < 1.0:
        n = len(train_ds)
        k = max(1, int(round(n * train_frac)))
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=k, replace=False)
        train_ds = Subset(train_ds, idx.tolist())

    encoder = TimerModel().to(device)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
        encoder.load_state_dict(sd, strict=True)
    encoder.eval()
    model = TimerLinearProbe(encoder, horizon).to(device)

    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    crit = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    best_val = float('inf'); best_state = None
    for ep in range(epochs):
        model.head.train()
        tr = 0.0; nb = 0
        for x_b, y_b, *_ in train_loader:
            x_b = x_b.to(device); y_b = y_b.to(device)
            pred = model(x_b)
            loss = crit(pred, y_b)
            opt.zero_grad(); loss.backward(); opt.step()
            tr += loss.item(); nb += 1
        tr /= max(nb, 1)

        model.head.eval()
        vl_mse = 0.0; vn = 0
        with torch.no_grad():
            for x_b, y_b, *_ in val_loader:
                x_b = x_b.to(device); y_b = y_b.to(device)
                pred = model(x_b)
                vl_mse += ((pred - y_b)**2).mean().item(); vn += 1
        vl_mse /= max(vn, 1)
        if vl_mse < best_val:
            best_val = vl_mse
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.head.state_dict().items()}
        if verbose:
            print(f'  ep{ep+1:>2}: train_mse={tr:.4f}  val_mse={vl_mse:.4f}  best={best_val:.4f}')

    if best_state is not None:
        model.head.load_state_dict(best_state)

    model.head.eval()
    mse_sum = 0.0; mae_sum = 0.0; n_el = 0
    with torch.no_grad():
        for x_b, y_b, *_ in test_loader:
            x_b = x_b.to(device); y_b = y_b.to(device)
            pred = model(x_b)
            mse_sum += ((pred - y_b)**2).sum().item()
            mae_sum += (pred - y_b).abs().sum().item()
            n_el += y_b.numel()
    test_mse = mse_sum / n_el
    test_mae = mae_sum / n_el

    if verbose:
        print(f'  -> {dataset_name} h={horizon} train_frac={train_frac}: '
              f'test_mse={test_mse:.4f} test_mae={test_mae:.4f}')
    return {'dataset': dataset_name, 'horizon': horizon,
            'test_mse': test_mse, 'test_mae': test_mae,
            'val_mse': best_val, 'train_frac': train_frac}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-ckpt', type=str, required=True)
    p.add_argument('--rho-cm-ckpt',   type=str, required=True)
    p.add_argument('--datasets',      type=str,
                   default='ETTh1,ETTh2,ETTm1,ETTm2,weather,exchange')
    p.add_argument('--horizons',      type=str, default='192')
    p.add_argument('--train-frac',    type=float, default=1.0)
    p.add_argument('--epochs',        type=int,   default=10)
    p.add_argument('--batch-size',    type=int,   default=256)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--out-json',      type=str,   required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  train_frac={args.train_frac}  epochs={args.epochs}')

    out = {'baseline': [], 'rho_cm': []}
    label_to_ckpt = {'baseline': args.baseline_ckpt, 'rho_cm': args.rho_cm_ckpt}

    datasets = args.datasets.split(',')
    horizons = [int(h) for h in args.horizons.split(',')]

    for label, ckpt in label_to_ckpt.items():
        print(f'\n===== {label}  ckpt={ckpt} =====')
        for ds in datasets:
            for h in horizons:
                try:
                    r = linear_probe_one(
                        ckpt, ds, h,
                        seed=args.seed, device=device,
                        train_frac=args.train_frac,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    )
                    out[label].append(r)
                except Exception as e:
                    print(f'  SKIP {ds} h={h}: {e}')

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, 'w'), indent=2)
    print(f'\nSaved: {args.out_json}')


if __name__ == '__main__':
    main()
