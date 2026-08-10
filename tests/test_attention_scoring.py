import torch

from scout_embedding.attention_scoring import (
    maximum_attention_distribution,
    scaled_dot_product_logits,
)


def test_maximum_attention_is_a_distribution_and_uses_strongest_row():
    layers = [
        torch.tensor([[[0.0, 2.0, 0.0]], [[3.0, 0.0, 0.0]]]),
        torch.tensor([[[0.0, 0.0, 4.0]], [[0.0, 0.0, 0.0]]]),
    ]
    scores = maximum_attention_distribution(layers)
    assert torch.allclose(scores.sum(), torch.tensor(1.0))
    assert scores.argmax().item() == 2
    assert (scores >= 0).all()


def test_scaled_dot_product_logits_shape():
    query = torch.randn(4, 2, 8)
    key = torch.randn(4, 7, 8)
    assert scaled_dot_product_logits(query, key).shape == (4, 2, 7)
