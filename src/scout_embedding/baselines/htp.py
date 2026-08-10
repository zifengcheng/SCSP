"""Independent HTP implementation compatible with SCOUT.

This module follows the algorithm in *Hierarchical Token Prepending* but does
not vendor code from the official repository. HTP owns the sentence-block
layout and hidden-state rewiring; SCOUT remains a separate token-selection and
readout operation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..input_construction import ScoutInput, SemanticChunkSpan

PST_TOKEN = "<PST>"
BPST_TOKEN = "<B-PST>"


def ensure_htp_tokens(tokenizer, model) -> tuple[int, int]:
    tokenizer.add_special_tokens({"additional_special_tokens": [PST_TOKEN, BPST_TOKEN]})
    pst = tokenizer.convert_tokens_to_ids(PST_TOKEN)
    bpst = tokenizer.convert_tokens_to_ids(BPST_TOKEN)
    embeddings = model.get_input_embeddings()
    required = max(pst, bpst) + 1
    if required > embeddings.num_embeddings:
        model.resize_token_embeddings(required, mean_resizing=True)
        embeddings = model.get_input_embeddings()
    with torch.no_grad():
        regular = embeddings.weight[: min(len(tokenizer) - 2, embeddings.num_embeddings)]
        mean = regular.float().mean(dim=0).to(embeddings.weight.dtype)
        embeddings.weight[pst].copy_(mean)
        embeddings.weight[bpst].copy_(mean)
    return pst, bpst


@dataclass(frozen=True)
class HTPLayout:
    model_input: ScoutInput
    pst_positions: tuple[int, ...]
    bpst_positions: tuple[int, ...]
    block_end_positions: tuple[int, ...]


def _group_block_indices(blocks: list[list[int]], budget: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    used = 0
    for index, block in enumerate(blocks):
        if current and used + len(block) > budget:
            groups.append(current)
            current, used = [], 0
        current.append(index)
        used += len(block)
    if current:
        groups.append(current)
    return groups


def build_htp_layout(
    sentence_blocks: list[list[int]],
    *,
    bos_token_id: int | None,
    pst_token_id: int,
    bpst_token_id: int,
    max_body_tokens: int,
    scout_chunk_size: int | None = None,
    compression_prompt_token_ids: list[int] | None = None,
    anchor_tokens: int = 1,
    leading_token_ids: list[int] | None = None,
) -> HTPLayout:
    retained: list[list[int]] = []
    used = 0
    for block in sentence_blocks:
        if used >= max_body_tokens:
            break
        kept = block[: max_body_tokens - used]
        if kept:
            retained.append(kept)
            used += len(kept)
    block_count = len(retained)
    ids: list[int] = []
    body_mask: list[bool] = []
    prompt_mask: list[bool] = []
    if leading_token_ids is not None:
        ids.extend(leading_token_ids)
        body_mask.extend([False] * len(leading_token_ids))
        prompt_mask.extend([False] * len(leading_token_ids))
    elif bos_token_id is not None:
        ids.append(bos_token_id)
        body_mask.append(False)
        prompt_mask.append(False)
    bpst_positions = tuple(range(len(ids), len(ids) + block_count))
    ids.extend([bpst_token_id] * block_count)
    body_mask.extend([False] * block_count)
    prompt_mask.extend([False] * block_count)
    common_prefix_end = len(ids)

    pst_positions: list[int] = []
    block_ends: list[int] = []
    chunks: list[SemanticChunkSpan] = []
    prompt_ids = compression_prompt_token_ids or []
    block_candidates: list[list[int]] = []
    block_context: list[list[int]] = []
    for block_index, block in enumerate(retained):
        pst_position = len(ids)
        pst_positions.append(pst_position)
        ids.append(pst_token_id)
        body_mask.append(False)
        prompt_mask.append(False)
        start = len(ids)
        ids.extend(block)
        body_mask.extend([True] * len(block))
        prompt_mask.extend([False] * len(block))
        candidates = list(range(start, len(ids)))
        block_candidates.append(candidates)
        block_context.append([bpst_positions[block_index], pst_position, *candidates])
        block_ends.append(len(ids) - 1)

    if scout_chunk_size is not None:
        if not prompt_ids:
            raise ValueError("HTP+SCOUT requires a non-empty compression prompt")
        for group in _group_block_indices(retained, scout_chunk_size):
            candidates = [
                position for index in group for position in block_candidates[index]
            ]
            context = [position for index in group for position in block_context[index]]
            prompt_start = len(ids)
            ids.extend(prompt_ids)
            body_mask.extend([False] * len(prompt_ids))
            prompt_mask.extend([True] * len(prompt_ids))
            prompt_end = len(ids)
            chunks.append(
                SemanticChunkSpan(
                    body_start=candidates[0],
                    body_end=candidates[-1] + 1,
                    anchor_start=prompt_end - anchor_tokens,
                    anchor_end=prompt_end,
                    prompt_start=prompt_start,
                    candidate_positions=tuple(candidates),
                    context_positions=tuple(context),
                )
            )
    model_input = ScoutInput(
        input_ids=tuple(ids),
        body_mask=tuple(body_mask),
        compression_prompt_mask=tuple(prompt_mask),
        chunks=tuple(chunks),
        common_prefix_end=common_prefix_end,
    )
    return HTPLayout(
        model_input=model_input,
        pst_positions=tuple(pst_positions),
        bpst_positions=bpst_positions,
        block_end_positions=tuple(block_ends),
    )


class HTPRewirer:
    """Copy each block-end hidden state into PST and B-PST slots."""

    def __init__(self, model, start: int, end: int):
        from ..attention_scoring import decoder_backbone

        self.layers = decoder_backbone(model).layers
        if not 0 <= start < end <= len(self.layers):
            raise ValueError("invalid HTP rewiring range")
        self.start, self.end = start, end
        self.layout: HTPLayout | None = None
        self.handles: list = []

    def set_layout(self, layout: HTPLayout) -> None:
        self.layout = layout

    def _hook(self, _module, args, kwargs):
        if self.layout is None:
            return args, kwargs
        hidden = args[0] if args else kwargs["hidden_states"]
        hidden = hidden.clone()
        ends = torch.tensor(self.layout.block_end_positions, device=hidden.device)
        pst = torch.tensor(self.layout.pst_positions, device=hidden.device)
        bpst = torch.tensor(self.layout.bpst_positions, device=hidden.device)
        source = hidden[0, ends].clone()
        hidden[0, pst] = source
        hidden[0, bpst] = source
        if args:
            return (hidden, *args[1:]), kwargs
        kwargs["hidden_states"] = hidden
        return args, kwargs

    def install(self) -> None:
        for layer in self.layers[self.start : self.end]:
            self.handles.append(
                layer.register_forward_pre_hook(self._hook, with_kwargs=True)
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.layout = None

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *_exc):
        self.remove()
