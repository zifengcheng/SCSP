import torch

from scout_embedding.token_selection import (
    select_above_uniform_attention,
    select_top_fraction,
)


def test_selection_compares_against_uniform_attention():
    probabilities = torch.tensor([0.05, 0.20, 0.30, 0.45])
    assert select_above_uniform_attention(probabilities, 1.0).tolist() == [
        False,
        False,
        True,
        True,
    ]


def test_uniform_ratio_keeps_at_least_one_token():
    probabilities = torch.full((4,), 0.25)
    selected = select_above_uniform_attention(probabilities, 10.0)
    assert selected.sum().item() == 1


def test_top_fraction_is_deterministic_in_count():
    selected = select_top_fraction(torch.arange(10.0), 0.3)
    assert selected.sum().item() == 3
