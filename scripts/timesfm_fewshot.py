"""TimesFM few-shot fine-tuning (linear probe or full FT).

Parallel to timer_fewshot.py. Loads a pretrained TimesFM PatchedTimeSeriesDecoder,
attaches a linear forecasting head from the last input-patch hidden state to the
horizon, and trains on a sub-sampled fraction of the target dataset.

Modes
-----
- (default) linear probe: encoder frozen, only head trained.
- --full-ft            : encoder + head jointly fine-tuned with
                         AdamW + cosine LR + grad clip (TimesFM official recipe).
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# TimesFM v1 source
_TIMESFM_SRC = _PROJECT_ROOT / 'rho_lib' / '_timesfm_v1_src'
if str(_TIMESFM_SRC) not in sys.path:
    sys.path.insert(0, str(_TIMESFM_SRC))
from timesfm_v1.pytorch_patched_decoder import PatchedTimeSeriesDecoder, TimesFMConfig

from rho_lib.data.forecast import prepare_forecast_datasets

CONTEXT_LEN = 512
PATCH_LEN   = 32
N_PATCHES   = CONTEXT_LEN // PATCH_LEN  # 16

DATA_DIR = _PROJECT_ROOT / 'data'


class TimesFMLinearProbe(nn.Module):
    """Frozen TimesFM encoder + linear head from last-patch hidden state to horizon.

    Channel-independent: (B, C, sl) -> reshape to (B*C, sl).
    Hidden state: use the last input position's hidden state (B*C, hidden_size).
    """

    def __init__(self, encoder: PatchedTimeSeriesDecoder, horizon: int, full_ft: bool = False):
        super().__init__()
        self.encoder = encoder
        self.horizon = horizon
        self.full_ft = full_ft
        self.head = nn.Linear(encoder.config.hidden_size, horizon, bias=True)
        for p in self.encoder.parameters():
            p.requires_grad = full_ft

    def _encode_hidden(self, input_ts: torch.Tensor) -> torch.Tensor:
        """input_ts (N, T) -> last-position hidden (N, hidden_size)."""
        N = input_ts.shape[0]
        padding = torch.zeros_like(input_ts)
        freq = torch.zeros(N, 1, dtype=torch.long, device=input_ts.device)
        def _run():
            model_input, patched_padding, stats, _ = self.encoder._preprocess_input(
                input_ts=input_ts, input_padding=padding)
            f_emb = self.encoder.freq_emb(freq)
            model_input = model_input + f_emb
            model_output = self.encoder.stacked_transformer(model_input, patched_padding)
            return model_output[:, -1, :], stats
        if self.full_ft:
            return _run()
        with torch.no_grad():
            return _run()

    def forward(self, x_bcs: torch.Tensor):
        # x_bcs: (B, C, sl)
        B, C, sl = x_bcs.shape
        assert sl == CONTEXT_LEN, f'sl={sl} but expected {CONTEXT_LEN}'
        x_flat = x_bcs.reshape(B * C, sl)
        # Per-sample z-score using context window
        mu  = x_flat.mean(dim=-1, keepdim=True)
        sig = x_flat.std(dim=-1, keepdim=True) + 1e-5
        x_norm = (x_flat - mu) / sig
        last_hidden, _ = self._encode_hidden(x_norm)  # (N, hidden)
        y = self.head(last_hidden)                     # (N, horizon)
        y = y * sig + mu                               # de-norm
        return y.reshape(B, C, self.horizon)


def build_timesfm_model() -> PatchedTimeSeriesDecoder:
    return PatchedTimeSeriesDecoder(TimesFMConfig())


def linear_probe_one(ckpt_path, dataset_name, horizon, *,
                     seed, device, train_frac=1.0,
                     epochs=10, batch_size=128, lr=1e-3,
                     context_length=CONTEXT_LEN, num_workers=0, verbose=True,
                     full_ft=False):
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

    encoder = build_timesfm_model().to(device)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
        encoder.load_state_dict(sd, strict=True)
    encoder.eval()
    model = TimesFMLinearProbe(encoder, horizon, full_ft=full_ft).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    n_steps_total = epochs * max(len(train_loader), 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps_total)
    crit = nn.MSELoss()

    best_val = float('inf'); best_state = None
    for ep in range(epochs):
        if model.full_ft:
            model.train()
        else:
            model.head.train()
        tr = 0.0; nb = 0
        for x_b, y_b, *_ in train_loader:
            x_b = x_b.to(device); y_b = y_b.to(device)
            pred = model(x_b)
            loss = crit(pred, y_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            tr += loss.item(); nb += 1
        tr /= max(nb, 1)

        model.head.eval()
        vl = 0.0; vn = 0
        with torch.no_grad():
            for x_b, y_b, *_ in val_loader:
                x_b = x_b.to(device); y_b = y_b.to(device)
                pred = model(x_b)
                vl += ((pred - y_b)**2).mean().item(); vn += 1
        vl /= max(vn, 1)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
        if verbose:
            print(f'  ep{ep+1:>2}: train_mse={tr:.4f}  val_mse={vl:.4f}  best={best_val:.4f}', flush=True)

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
    test_mse = mse_sum / n_el; test_mae = mae_sum / n_el
    if verbose:
        print(f'  -> {dataset_name} h={horizon} train_frac={train_frac}: '
              f'test_mse={test_mse:.4f} test_mae={test_mae:.4f}', flush=True)
    return {'dataset': dataset_name, 'horizon': horizon,
            'test_mse': test_mse, 'test_mae': test_mae,
            'val_mse': best_val, 'train_frac': train_frac}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-ckpt', type=str, required=True)
    p.add_argument('--rho-cm-ckpt',   type=str, required=True)
    p.add_argument('--datasets',      type=str,
                   default='ETTh1,ETTh2,ETTm1,ETTm2,weather,exchange')
    p.add_argument('--horizons',      type=str, default='96')
    p.add_argument('--train-frac',    type=float, default=1.0)
    p.add_argument('--epochs',        type=int,   default=10)
    p.add_argument('--batch-size',    type=int,   default=128)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--full-ft',       action='store_true',
                   help='Full fine-tuning (unfreeze encoder)')
    p.add_argument('--out-json',      type=str,   required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  train_frac={args.train_frac}  epochs={args.epochs}', flush=True)

    out = {'baseline': [], 'rho_cm': []}
    label_to_ckpt = {'baseline': args.baseline_ckpt, 'rho_cm': args.rho_cm_ckpt}

    datasets = args.datasets.split(',')
    horizons = [int(h) for h in args.horizons.split(',')]

    for label, ckpt in label_to_ckpt.items():
        print(f'\n===== {label}  ckpt={ckpt} =====', flush=True)
        for ds in datasets:
            for h in horizons:
                try:
                    r = linear_probe_one(
                        ckpt, ds, h,
                        seed=args.seed, device=device,
                        train_frac=args.train_frac,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                        full_ft=args.full_ft,
                    )
                    out[label].append(r)
                except Exception as e:
                    print(f'  SKIP {ds} h={h}: {e}', flush=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, 'w'), indent=2)
    print(f'\nSaved: {args.out_json}', flush=True)


if __name__ == '__main__':
    main()
