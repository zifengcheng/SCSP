from scout_embedding.config import ExperimentConfig, ModelConfig, ScoutConfig
from scout_embedding.modeling import LongContextEmbeddingEncoder


class TinyTokenizer:
    bos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == '\nBrief summary:"':
            return [90, 91]
        return [10 + index for index, _character in enumerate(text)]


def make_encoder() -> LongContextEmbeddingEncoder:
    encoder = LongContextEmbeddingEncoder.__new__(LongContextEmbeddingEncoder)
    encoder.config = ExperimentConfig(
        method="echo_scout",
        model=ModelConfig(max_body_tokens=8),
        scout=ScoutConfig(chunk_size=3, chunking="fixed"),
    )
    encoder.tokenizer = TinyTokenizer()
    return encoder


def test_echo_scout_preserves_the_complete_parent_input_prefix():
    encoder = make_encoder()
    parent = encoder._echo_layout("abcdef")
    combined = encoder._scout_input("abcdef", parent="echo")

    parent_length = len(parent.input_ids)
    assert combined.input_ids[:parent_length] == parent.input_ids
    assert combined.body_mask[:parent_length] == parent.body_mask
    assert not any(combined.compression_prompt_mask[:parent_length])
    assert all(span.prompt_start >= parent_length for span in combined.chunks)


def test_scout_keeps_only_real_document_tokens_in_the_body_mask():
    encoder = make_encoder()
    model_input = encoder._scout_input("abcdef", parent="vanilla")

    retained = [
        token
        for token, is_body in zip(model_input.input_ids, model_input.body_mask)
        if is_body
    ]
    assert retained == [10, 11, 12, 13, 14, 15]
    assert all(
        not model_input.body_mask[position]
        for chunk in model_input.chunks
        for position in range(chunk.prompt_start, chunk.anchor_end)
    )
