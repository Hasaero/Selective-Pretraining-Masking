"""Unified UTSD pretraining dataset.

Memory-light by default: stores one (C, T) array per series, slices windows
lazily in __getitem__. Optional per-series z-score (applied once at __init__)
and `max_windows` cap (Timer-style streaming early-stop).
"""
import time as _time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class UTSDPretrainDataset(Dataset):
    """
    Args:
        hf_dataset             : HF Dataset (load_from_disk result)
        seq_len                : window length in timesteps
        stride                 : window stride
        max_series             : cap on number of source series (None = all)
        max_windows            : cap on number of training windows (None = all)
        standardize_per_series : if True, per-series z-score the (C, T) array
                                 once at __init__ (MOMENT/Timer behaviour).
                                 Default False (raw values; backbone handles its own norm)
                                 because its internal scaler handles it.
        skip_short             : if True, drop series with T < seq_len.
                                 Default True (otherwise they yield no windows
                                 anyway).

    Returns from __getitem__:
        (window_idx, window_tensor)  where window_tensor shape = (C, seq_len)
    """

    def __init__(
        self,
        hf_dataset,
        seq_len: int,
        stride: int,
        *,
        max_series: int | None = None,
        max_windows: int | None = None,
        standardize_per_series: bool = False,
        skip_short: bool = True,
        verbose: bool = True,
    ):
        self.seq_len = seq_len
        self.stride  = stride

        if verbose:
            print(f'[Data] streaming UTSD '
                  f'(seq_len={seq_len}, stride={stride}, '
                  f'max_series={max_series}, max_windows={max_windows}, '
                  f'standardize={standardize_per_series})...', flush=True)

        # One contiguous (C, T) array per series; (sidx, start) per window.
        self._series:        list[np.ndarray]              = []
        self._series_domain: list[str]                     = []
        self._index:         list[tuple[int, int]]         = []

        n_series, n_eligible, n_skipped, n_rows = 0, 0, 0, 0
        cur_key, cur_buf = None, []
        lengths = []
        t0 = _time.time()
        done = False

        def _flush(buf, key=None) -> bool:
            """Materialise one series → optionally z-score → window indices.
            Returns True if we should stop (max_windows reached)."""
            nonlocal n_eligible, n_skipped
            if not buf:
                return False
            domain = (key.split('_', 1)[0] if key else 'Unknown')
            ch_list = sorted(buf, key=lambda x: x[0])
            arr = np.stack([c[1] for c in ch_list], axis=0)            # (C, T)
            T = arr.shape[1]
            lengths.append(T)
            if skip_short and T < seq_len:
                n_skipped += 1
                return False
            if standardize_per_series:
                mu  = arr.mean(axis=1, keepdims=True)
                sig = arr.std(axis=1, keepdims=True) + 1e-8
                arr = (arr - mu) / sig
            n_eligible += 1
            sidx = len(self._series)
            self._series.append(arr)
            self._series_domain.append(domain)
            for s in range(0, T - seq_len + 1, stride):
                self._index.append((sidx, s))
                if max_windows is not None and len(self._index) >= max_windows:
                    return True
            return False

        for batch in hf_dataset.iter(batch_size=1000):
            if done:
                break
            for iid, target in zip(batch['item_id'], batch['target']):
                parts = iid.rsplit('_', 1)
                key = parts[0]
                if key != cur_key and cur_key is not None:
                    n_series += 1
                    full = _flush(cur_buf, key=cur_key)
                    cur_buf = []
                    if full:
                        done = True
                        break
                    if max_series is not None and n_series >= max_series:
                        done = True
                        break
                cur_key = key
                cur_buf.append((
                    int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0,
                    np.array(target, dtype=np.float32),
                ))
                n_rows += 1
                if verbose and n_rows % 10000 == 0:
                    print(f'[Data] grouping... {n_rows} rows, '
                          f'{n_series} series done, '
                          f'{len(self._index)} windows', flush=True)

        # Flush trailing series if iteration ended cleanly
        if not done and cur_buf:
            _flush(cur_buf, key=cur_key)
            n_series += 1

        if verbose:
            if lengths:
                arr = np.array(lengths)
                print(f'[Data] series-length stats (seen): '
                      f'min={arr.min()} median={int(np.median(arr))} '
                      f'max={arr.max()} mean={arr.mean():.0f}', flush=True)
            print(f'[UTSDPretrainDataset] {len(self._index)} windows '
                  f'from {n_eligible} eligible series '
                  f'(skipped {n_skipped} short, {n_series} total seen, '
                  f'{_time.time()-t0:.1f}s)', flush=True)

    # ── Accessors ─────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        sidx, start = self._index[idx]
        window = self._series[sidx][:, start:start + self.seq_len]
        return idx, torch.from_numpy(np.ascontiguousarray(window))

    def domain_of(self, idx: int) -> str:
        """Return the domain prefix of the series owning window ."""
        sidx, _ = self._index[idx]
        return self._series_domain[sidx]

    def channel_counts_per_series(self) -> list[int]:
        """One entry per *series* — used for percentile-based caps."""
        return [arr.shape[0] for arr in self._series]

    def channel_counts_per_window(self) -> list[int]:
        """One entry per *window* — preserves MOMENT/Timer's older behaviour
        where percentile is computed over windows (long series weighted more)."""
        return [self._series[sidx].shape[0] for sidx, _ in self._index]


def compute_channel_cap(dataset: UTSDPretrainDataset,
                        percentile: float = 99.0,
                        per: str = 'window') -> int:
    """Compute a percentile-based channel cap.

    `per='window'` matches MOMENT/Timer (legacy behaviour) — windows from long
    series are weighted more. `per='series'` is unbiased per-series.
    """
    if per == 'window':
        c_sizes = dataset.channel_counts_per_window()
    elif per == 'series':
        c_sizes = dataset.channel_counts_per_series()
    else:
        raise ValueError(f'per must be "window" or "series", got {per!r}')
    return int(np.percentile(c_sizes, percentile))


def make_collate_fn(seq_len: int, channel_cap: int | None = None):
    """Pad channels to max-C in batch. If `channel_cap` is set, randomly
    subsample channels above the cap (Timer behaviour, prevents SDPA grid
    overflow from heavy multivariate series like traffic with 862 channels)."""
    def collate_fn(batch):
        indices, tensors = zip(*batch)
        indices = torch.tensor(indices, dtype=torch.long)
        if channel_cap is not None:
            capped = []
            for t in tensors:
                c = t.shape[0]
                if c > channel_cap:
                    sel = torch.randperm(c)[:channel_cap]
                    t = t[sel]
                capped.append(t)
            tensors = capped
        max_c = max(t.shape[0] for t in tensors)
        B = len(tensors)
        x = torch.zeros(B, max_c, seq_len)
        ch_mask = torch.zeros(B, max_c, dtype=torch.bool)
        for i, t in enumerate(tensors):
            c = t.shape[0]
            x[i, :c]       = t
            ch_mask[i, :c] = True
        return indices, x, ch_mask
    return collate_fn


def load_utsd(path: Path | str, max_series: int | None = None):
    """HF dataset load + max_series pre-slice (saves the iter() scan when
    we only want the first N series)."""
    from datasets import load_from_disk
    hf_ds = load_from_disk(str(path))
    if max_series is not None:
        n_rows = min(max_series * 20, len(hf_ds))
        hf_ds  = hf_ds.select(range(n_rows))
    return hf_ds


class UTSDPretrainDatasetUnivariate(Dataset):
    """OpenLTM-style SOM (Single-variate Origin Mode): unroll every multivariate
    series into independent univariate windows.

    Each item is a single (1, seq_len) window from one (series, channel) pair.
    This matches Timer's official pretraining and avoids the channel-cap
    information loss required by the multivariate format.

    Index entries: (series_idx, channel_idx, start_offset). When
    `random_offset=True`, the stored start is ignored at fetch time and a
    fresh uniform random start in [0, T - seq_len] is drawn per __getitem__.
    """
    def __init__(
        self,
        hf_dataset,
        seq_len: int,
        stride: int,
        *,
        max_series: int | None = None,
        max_windows: int | None = None,
        standardize_per_series: bool = False,
        skip_short: bool = True,
        verbose: bool = True,
    ):
        import time as _time
        self.seq_len = seq_len
        self.stride  = stride

        if verbose:
            print(f'[Data] streaming UTSD (univariate SOM, '
                  f'seq_len={seq_len}, stride={stride}, '
                  f'max_series={max_series}, max_windows={max_windows}, '
                  f'standardize={standardize_per_series})...', flush=True)

        self._series:        list[np.ndarray] = []   # (C, T) arrays
        self._series_domain: list[str]        = []
        # univariate index: (series_idx, channel_idx, start)
        self._index:         list[tuple[int, int, int]] = []

        n_series, n_eligible, n_skipped, n_rows = 0, 0, 0, 0
        cur_key, cur_buf = None, []
        lengths = []
        t0 = _time.time()
        done = False

        def _flush(buf, key=None) -> bool:
            nonlocal n_eligible, n_skipped
            if not buf:
                return False
            domain = (key.split('_', 1)[0] if key else 'Unknown')
            ch_list = sorted(buf, key=lambda x: x[0])
            arr = np.stack([c[1] for c in ch_list], axis=0)            # (C, T)
            C, T = arr.shape
            lengths.append(T)
            if skip_short and T < seq_len:
                n_skipped += 1
                return False
            if standardize_per_series:
                mu  = arr.mean(axis=1, keepdims=True)
                sig = arr.std(axis=1, keepdims=True) + 1e-8
                arr = (arr - mu) / sig
            n_eligible += 1
            sidx = len(self._series)
            self._series.append(arr)
            self._series_domain.append(domain)
            # Each (channel, start) becomes a univariate window
            for c in range(C):
                for s in range(0, T - seq_len + 1, stride):
                    self._index.append((sidx, c, s))
                    if max_windows is not None and len(self._index) >= max_windows:
                        return True
            return False

        for batch in hf_dataset.iter(batch_size=1000):
            if done:
                break
            for iid, target in zip(batch['item_id'], batch['target']):
                parts = iid.rsplit('_', 1)
                key = parts[0]
                if key != cur_key and cur_key is not None:
                    n_series += 1
                    full = _flush(cur_buf, key=cur_key)
                    cur_buf = []
                    if full:
                        done = True
                        break
                    if max_series is not None and n_series >= max_series:
                        done = True
                        break
                cur_key = key
                cur_buf.append((
                    int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0,
                    np.array(target, dtype=np.float32),
                ))
                n_rows += 1
                if verbose and n_rows % 10000 == 0:
                    print(f'[Data] grouping... {n_rows} rows, '
                          f'{n_series} series done, '
                          f'{len(self._index)} windows', flush=True)

        if not done and cur_buf:
            _flush(cur_buf, key=cur_key)
            n_series += 1

        if verbose:
            print(f'[UTSDPretrainDatasetUnivariate] {len(self._index)} '
                  f'univariate windows from {n_eligible} series '
                  f'(skipped {n_skipped} short, {n_series} total, '
                  f'{_time.time()-t0:.1f}s)', flush=True)

    def __len__(self):
        return len(self._index)

    # Set externally (default False). When True, __getitem__ draws a fresh
    # random start within [0, T-seq_len] each call (uni2ts-style sampling).
    random_offset: bool = False

    def __getitem__(self, idx):
        sidx, cidx, start = self._index[idx]
        if getattr(self, 'random_offset', False):
            T = self._series[sidx].shape[1]
            max_start = T - self.seq_len
            if max_start > 0:
                import random as _random
                start = _random.randint(0, max_start)
            else:
                start = 0
        window = self._series[sidx][cidx, start:start + self.seq_len]  # (sl,)
        return idx, torch.from_numpy(np.ascontiguousarray(window))

    def domain_of(self, idx: int) -> str:
        sidx, _, _ = self._index[idx]
        return self._series_domain[sidx]

    # ----- Shims so ref builders that expect MV dataset still work -----
    def channel_counts_per_series(self) -> list[int]:
        return [1] * len(self._series)

    def channel_counts_per_window(self) -> list[int]:
        return [1] * len(self._index)


def make_collate_fn_univariate(seq_len: int):
    """Collate function for univariate (SOM) batches.

    Each sample is (sl,); the batch becomes (B, 1, sl) — channel dim = 1.
    No channel padding, no channel cap (each item is already univariate).
    """
    def collate_fn(batch):
        indices, tensors = zip(*batch)
        indices = torch.tensor(indices, dtype=torch.long)
        # Stack univariate windows: (B, sl) -> (B, 1, sl)
        x = torch.stack(tensors, dim=0).unsqueeze(1)        # (B, 1, sl)
        ch_mask = torch.ones(x.shape[0], 1, dtype=torch.bool)
        return indices, x, ch_mask
    return collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# Disk cache for univariate datasets
# ─────────────────────────────────────────────────────────────────────────────
def _cache_key(seq_len: int, stride: int, standardize: bool,
               max_series: int | None, max_windows: int | None) -> str:
    """Build a unique cache filename from dataset construction args."""
    parts = [f'sl{seq_len}', f'str{stride}',
             f'zsc{int(standardize)}']
    if max_series is not None:
        parts.append(f'ms{max_series}')
    if max_windows is not None:
        parts.append(f'mw{max_windows}')
    return 'utsd_univ__' + '_'.join(parts) + '.pkl'


def load_or_build_cached_univariate(
    utsd_path: Path | str,
    seq_len: int,
    stride: int,
    *,
    standardize_per_series: bool,
    max_series: int | None = None,
    max_windows: int | None = None,
    cache_dir: Path | str | None = None,
    random_offset: bool = False,
):
    """Load univariate dataset from pickle cache if available, else build and save.

    Cache key encodes (seq_len, stride, standardize, max_series, max_windows)
    so different configs get separate files. Default cache_dir is
    <utsd_path>/../cache/.
    """
    import pickle as _pkl
    import time as _time
    utsd_path = Path(utsd_path)
    if cache_dir is None:
        cache_dir = utsd_path.parent / 'cache'
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / _cache_key(seq_len, stride,
                                         standardize_per_series,
                                         max_series, max_windows)

    if cache_path.exists():
        print(f'[Cache] Loading univariate dataset from {cache_path} ...')
        t0 = _time.time()
        with open(cache_path, 'rb') as f:
            ds = _pkl.load(f)
        print(f'[Cache] Loaded {len(ds)} windows in {_time.time()-t0:.1f}s')
        ds.random_offset = random_offset
        if random_offset:
            print(f'[Cache] random_offset=True (each __getitem__ draws uniform start)')
        return ds

    print(f'[Cache] No cache found at {cache_path} — building...')
    hf_ds = load_utsd(utsd_path, max_series=max_series)
    ds = UTSDPretrainDatasetUnivariate(
        hf_ds, seq_len=seq_len, stride=stride,
        max_series=max_series, max_windows=max_windows,
        standardize_per_series=standardize_per_series,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f'[Cache] Saving to {cache_path} ...')
    t0 = _time.time()
    with open(cache_path, 'wb') as f:
        _pkl.dump(ds, f, protocol=_pkl.HIGHEST_PROTOCOL)
    sz_mb = cache_path.stat().st_size / (1 << 20)
    print(f'[Cache] Saved {sz_mb:.0f} MB in {_time.time()-t0:.1f}s')
    ds.random_offset = random_offset
    if random_offset:
        print(f'[Cache] random_offset=True (each __getitem__ draws uniform start)')
    return ds
