"""Forecasting eval datasets (ETT, electricity, traffic, weather, exchange,
ILI, ECL, solar). Self-contained — no external dependency on gracm."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DATASET_FILES = {
    'electricity': Path('electricity/electricity.csv'),
    'traffic':     Path('traffic/traffic.csv'),
    'weather':     Path('weather/weather.csv'),
    'etth1':       Path('ETT-small/ETTh1.csv'),
    'etth2':       Path('ETT-small/ETTh2.csv'),
    'ettm1':       Path('ETT-small/ETTm1.csv'),
    'ettm2':       Path('ETT-small/ETTm2.csv'),
    'exchange':    Path('Exchange/Exchange.csv'),
    'ili':         Path('ILI/ILI.csv'),
    'solar':       Path('Solar/solar_AL.txt'),
    'ecl':         Path('ECL/ECL.csv'),
}


def load_forecast_csv(path: Path):
    import pandas as pd
    if path.suffix == '.txt':
        df = pd.read_csv(path, sep=r'\s+', header=None)
        timestamps = None
    else:
        df = pd.read_csv(path)
        time_cols = {'date', 'Date', 'DATE', 'timestamp', 'Timestamp', 'time', 'Time'}
        date_col = next((c for c in df.columns if c in time_cols), None)
        if date_col is not None:
            timestamps = pd.to_datetime(df[date_col])
            df = df.drop(columns=[date_col])
        else:
            timestamps = None
    df = df.apply(pd.to_numeric, errors='coerce').ffill().bfill().fillna(0.0)
    return df.values.astype(np.float32), timestamps


class ForecastWindowDataset(Dataset):
    """Yields (enc, dec, mask) where enc=(C, sl), dec=(C, fh)."""
    def __init__(self, data, sequence_length, forecast_horizon, stride=1):
        self.data = data
        self.sl = sequence_length
        self.fh = forecast_horizon
        max_start = data.shape[0] - sequence_length - forecast_horizon
        self.starts = list(range(0, max_start + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        enc = self.data[s:s + self.sl].T
        dec = self.data[s + self.sl:s + self.sl + self.fh].T
        mask = np.ones(self.sl, dtype=np.float32)
        return (torch.from_numpy(enc).float(),
                torch.from_numpy(dec).float(),
                torch.from_numpy(mask).float())


def prepare_forecast_datasets(dataset_name: str, data_root: Path,
                              sequence_length: int, forecast_horizon: int,
                              stride: int = 1):
    """Return (train, val, test) ForecastWindowDataset triple, normalised by
    train-split per-channel mean/std. Honours the standard ETT split rules."""
    key = dataset_name.lower()
    if key not in DATASET_FILES:
        raise ValueError(f'Unsupported dataset {dataset_name}. '
                         f'Available: {sorted(DATASET_FILES)}')
    path = (Path(data_root) / DATASET_FILES[key]).resolve()
    data, _ = load_forecast_csv(path)
    total = len(data)
    seq_len = sequence_length

    if key in ('ettm1', 'ettm2'):
        b1s = [0, 12*30*24*4 - seq_len, 12*30*24*4 + 4*30*24*4 - seq_len]
        b2s = [12*30*24*4, 12*30*24*4 + 4*30*24*4, 12*30*24*4 + 8*30*24*4]
    elif key in ('etth1', 'etth2'):
        b1s = [0, 12*30*24 - seq_len, 12*30*24 + 4*30*24 - seq_len]
        b2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]
    else:
        n_train = int(total * 0.7)
        n_test  = int(total * 0.2)
        n_val   = total - n_train - n_test
        b1s = [0, n_train - seq_len, total - n_test - seq_len]
        b2s = [n_train, n_train + n_val, total]

    train_raw = data[max(0, b1s[0]):min(total, b2s[0])]
    val_raw   = data[max(0, b1s[1]):min(total, b2s[1])]
    test_raw  = data[max(0, b1s[2]):min(total, b2s[2])]

    mean = train_raw.mean(axis=0, keepdims=True)
    std  = train_raw.std(axis=0,  keepdims=True) + 1e-8
    return (
        ForecastWindowDataset((train_raw - mean) / std, seq_len, forecast_horizon, stride),
        ForecastWindowDataset((val_raw   - mean) / std, seq_len, forecast_horizon, stride),
        ForecastWindowDataset((test_raw  - mean) / std, seq_len, forecast_horizon, stride),
    )
