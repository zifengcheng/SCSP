import torch
from transformers import MistralConfig, MistralModel

from scout_embedding.attention_scoring import SemanticCompressionAttentionCollector
from scout_embedding.baselines.htp import HTPRewirer, build_htp_layout
from scout_embedding.chunking import TokenChunk
from scout_embedding.input_construction import (
    build_prompt_isolation_mask,
    build_scout_input,
    continuous_position_ids,
)


def test_sparse_collector_runs_on_a_tiny_gqa_decoder():
    model = MistralModel(
        MistralConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
        )
    ).eval()
    layout = build_scout_input(
        [TokenChunk((4, 5, 6), 1), TokenChunk((7, 8), 1)],
        bos_token_id=1,
        compression_prompt_token_ids=[9, 10],
        anchor_tokens=1,
    )
    input_ids = torch.tensor([layout.input_ids])
    positions = continuous_position_ids(input_ids.shape[1], "cpu")
    attention = build_prompt_isolation_mask(
        layout, scope="local", dtype=torch.float32, device="cpu"
    )
    collector = SemanticCompressionAttentionCollector(model, layout.chunks, positions)
    with collector:
        model(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=positions,
            use_cache=False,
        )
        scores = collector.token_scores()
    assert [score.numel() for score in scores] == [3, 2]
    assert all(torch.allclose(score.sum(), torch.tensor(1.0)) for score in scores)


def test_htp_and_scout_share_one_forward_without_pooling_prompts():
    model = MistralModel(
        MistralConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
        )
    ).eval()
    htp = build_htp_layout(
        [[4, 5], [6, 7]],
        bos_token_id=1,
        pst_token_id=2,
        bpst_token_id=3,
        max_body_tokens=16,
        scout_chunk_size=2,
        compression_prompt_token_ids=[9, 10],
    )
    layout = htp.model_input
    input_ids = torch.tensor([layout.input_ids])
    positions = continuous_position_ids(input_ids.shape[1], "cpu")
    attention = build_prompt_isolation_mask(
        layout, scope="local", dtype=torch.float32, device="cpu"
    )
    collector = SemanticCompressionAttentionCollector(model, layout.chunks, positions)
    with HTPRewirer(model, 0, 1) as rewiring, collector:
        rewiring.set_layout(htp)
        model(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=positions,
            use_cache=False,
        )
        scores = collector.token_scores()
    assert [score.numel() for score in scores] == [2, 2]
    assert sum(layout.body_mask) == 4
