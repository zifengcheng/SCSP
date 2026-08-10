"""Mask-aware embedding pooling."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError("expected hidden [batch, length, dim] and mask [batch, length]")
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1)
    return (hidden * weights).sum(dim=1) / denominator


def normalized_masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.normalize(masked_mean(hidden, mask).float(), p=2, dim=-1)
