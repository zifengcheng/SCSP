import torch

from scout_embedding.chunking import TokenChunk
from scout_embedding.input_construction import (
    build_prompt_isolation_mask,
    build_scout_input,
)


def make_layout():
    return build_scout_input(
        [TokenChunk((1, 2), 1), TokenChunk((3, 4), 1)],
        bos_token_id=0,
        compression_prompt_token_ids=[8, 9],
        anchor_tokens=1,
    )


def test_local_scope_preserves_text_causality_and_hides_prompts():
    mask = build_prompt_isolation_mask(
        make_layout(), scope="local", dtype=torch.float32, device="cpu"
    )[0, 0]
    assert mask[5, 2] == 0  # later text sees earlier real text
    assert mask[5, 3] < 0  # later text cannot read an earlier prompt
    assert mask[8, 5] == 0  # prompt 2 reads chunk 2
    assert mask[8, 1] < 0  # prompt 2 cannot read chunk 1
    assert mask[8, 3] < 0  # prompt slots are mutually hidden


def test_cumulative_differs_only_in_prompt_visibility():
    layout = make_layout()
    direct = build_prompt_isolation_mask(
        layout, scope="local", dtype=torch.float32, device="cpu"
    )[0, 0]
    cumulative = build_prompt_isolation_mask(
        layout, scope="cumulative", dtype=torch.float32, device="cpu"
    )[0, 0]
    body = torch.tensor(layout.body_mask)
    assert torch.equal(direct[body], cumulative[body])
    assert direct[8, 1] < 0 and cumulative[8, 1] == 0
