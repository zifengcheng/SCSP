import torch
from torch import nn

from scout_embedding.baselines.htp import HTPRewirer, build_htp_layout


class IdentityLayer(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([IdentityLayer()])


def test_htp_rewiring_copies_block_ends_to_both_slot_types():
    model = TinyBackbone()
    layout = build_htp_layout(
        [[10, 11], [12]],
        bos_token_id=1,
        pst_token_id=2,
        bpst_token_id=3,
        max_body_tokens=20,
    )
    hidden = torch.arange(len(layout.model_input.input_ids), dtype=torch.float32).view(
        1, -1, 1
    )
    with HTPRewirer(model, 0, 1) as rewiring:
        rewiring.set_layout(layout)
        output = model.layers[0](hidden)
    for pst, bpst, end in zip(
        layout.pst_positions, layout.bpst_positions, layout.block_end_positions
    ):
        assert output[0, pst].item() == hidden[0, end].item()
        assert output[0, bpst].item() == hidden[0, end].item()


def test_htp_scout_layout_excludes_structural_tokens_from_body_mask():
    parent = build_htp_layout(
        [[10, 11], [12, 13]],
        bos_token_id=1,
        pst_token_id=2,
        bpst_token_id=3,
        max_body_tokens=20,
    )
    layout = build_htp_layout(
        [[10, 11], [12, 13]],
        bos_token_id=1,
        pst_token_id=2,
        bpst_token_id=3,
        max_body_tokens=20,
        scout_chunk_size=3,
        compression_prompt_token_ids=[8, 9],
    )
    assert sum(layout.model_input.body_mask) == 4
    assert (
        layout.model_input.input_ids[: len(parent.model_input.input_ids)]
        == parent.model_input.input_ids
    )
    assert layout.pst_positions == parent.pst_positions
    assert layout.bpst_positions == parent.bpst_positions
    assert min(chunk.prompt_start for chunk in layout.model_input.chunks) > max(
        layout.block_end_positions
    )
    assert all(not layout.model_input.body_mask[index] for index in layout.pst_positions)
    assert all(not layout.model_input.body_mask[index] for index in layout.bpst_positions)
