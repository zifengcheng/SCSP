from scout_embedding.chunking import fixed_token_chunks, truncate_chunks


def test_fixed_chunks_have_no_overlap_or_duplication():
    chunks = fixed_token_chunks(list(range(10)), 4)
    assert [list(chunk.token_ids) for chunk in chunks] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]


def test_truncation_caps_body_tokens_only():
    chunks = fixed_token_chunks(list(range(12)), 5)
    retained = truncate_chunks(chunks, 8)
    assert [token for chunk in retained for token in chunk.token_ids] == list(range(8))
