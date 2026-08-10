"""Token-selection rules kept separate from attention scoring."""

from __future__ import annotations

import torch


def select_above_uniform_attention(
    probabilities: torch.Tensor,
    ratio: float,
    *,
    min_keep: int = 1,
) -> torch.Tensor:
    """Keep tokens whose attention is at least ``ratio / token_count``.

    The reference point is a uniform distribution over the current score
    window. ``ratio=1`` therefore keeps tokens receiving at least the window's
    mean probability.
    """
    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("probabilities must be a non-empty vector")
    if ratio < 0 or min_keep < 1:
        raise ValueError("ratio must be non-negative and min_keep positive")
    threshold = ratio / probabilities.numel()
    selected = probabilities >= threshold
    needed = min(min_keep, probabilities.numel()) - int(selected.sum().item())
    if needed > 0:
        selected[torch.topk(probabilities, k=min_keep).indices] = True
    return selected


def select_top_fraction(
    scores: torch.Tensor,
    fraction: float,
    *,
    min_keep: int = 1,
) -> torch.Tensor:
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty vector")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    keep = max(min_keep, round(fraction * scores.numel()))
    keep = min(keep, scores.numel())
    selected = torch.zeros_like(scores, dtype=torch.bool)
    selected[torch.topk(scores, k=keep).indices] = True
    return selected
