"""Checkpoint save / metrics dump."""
import json
from pathlib import Path

import torch


def save_epoch(out_dir: Path, epoch: int, state_dict, loss: float,
               best_loss: float) -> float:
    """Save `epoch{epoch:03d}.pt` and update `best.pt` if loss improved.
    Returns the new best_loss."""
    out_dir = Path(out_dir)
    ckpt = {'epoch': epoch, 'state_dict': state_dict, 'loss': loss}
    torch.save(ckpt, out_dir / f'epoch{epoch:03d}.pt')
    if loss < best_loss:
        best_loss = loss
        torch.save(ckpt, out_dir / 'best.pt')
    return best_loss


def dump_metrics(out_dir: Path, metrics: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
