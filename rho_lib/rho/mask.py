"""RHO mask computation — shared across moirai/moment/timer.

`rho = current_loss - ref_loss` over an arbitrary tensor shape; we keep entries
above the bottom drop_pct% threshold (i.e. drop the easiest fraction, where
`current` is closest to or below `ref`).
"""
import torch

_MAX_QUANTILE_ELEMS = 1 << 24  # torch.quantile hard limit


def safe_quantile(t: torch.Tensor, q: float) -> torch.Tensor:
    """torch.quantile that subsamples if input exceeds the hard limit."""
    if t.numel() > _MAX_QUANTILE_ELEMS:
        idx = torch.randperm(t.numel(), device=t.device)[:_MAX_QUANTILE_ELEMS]
        return torch.quantile(t[idx].float(), q)
    return torch.quantile(t.float(), q)


def compute_rho_mask(rho: torch.Tensor,
                     valid_mask: torch.Tensor,
                     drop_pct: float) -> torch.Tensor:
    """Returns bool mask — True = kept (above threshold). If thresholding
    produces an empty mask (degenerate), falls back to keeping all valid
    entries to avoid zero-denominator loss."""
    flat_rho  = rho[valid_mask]
    threshold = safe_quantile(flat_rho, drop_pct / 100.0)
    mask = (rho >= threshold) & valid_mask
    return mask if mask.any() else valid_mask
