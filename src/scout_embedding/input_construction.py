"""SCOUT sequence construction and prompt-isolated attention masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .chunking import TokenChunk


@dataclass(frozen=True)
class SemanticChunkSpan:
    """Physical positions owned by one semantic chunk and its prompt."""

    body_start: int
    body_end: int
    anchor_start: int
    anchor_end: int
    prompt_start: int
    candidate_positions: tuple[int, ...] = ()
    context_positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class ScoutInput:
    """A tokenized SCOUT document with explicit semantic chunk boundaries."""

    input_ids: tuple[int, ...]
    body_mask: tuple[bool, ...]
    compression_prompt_mask: tuple[bool, ...]
    chunks: tuple[SemanticChunkSpan, ...]
    common_prefix_end: int


def build_scout_input(
    chunks: list[TokenChunk],
    *,
    bos_token_id: int | None,
    compression_prompt_token_ids: list[int],
    anchor_tokens: int = 1,
    leading_token_ids: list[int] | None = None,
) -> ScoutInput:
    if not compression_prompt_token_ids:
        raise ValueError("the semantic compression prompt cannot be empty")
    if not 1 <= anchor_tokens <= len(compression_prompt_token_ids):
        raise ValueError("anchor_tokens must fit inside the compression prompt")

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
    common_prefix_end = len(ids)

    spans: list[SemanticChunkSpan] = []
    for chunk in chunks:
        body_start = len(ids)
        ids.extend(chunk.token_ids)
        body_mask.extend([True] * len(chunk.token_ids))
        prompt_mask.extend([False] * len(chunk.token_ids))
        body_end = len(ids)
        prompt_start = len(ids)
        ids.extend(compression_prompt_token_ids)
        body_mask.extend([False] * len(compression_prompt_token_ids))
        prompt_mask.extend([True] * len(compression_prompt_token_ids))
        prompt_end = len(ids)
        spans.append(
            SemanticChunkSpan(
                body_start=body_start,
                body_end=body_end,
                anchor_start=prompt_end - anchor_tokens,
                anchor_end=prompt_end,
                prompt_start=prompt_start,
                candidate_positions=tuple(range(body_start, body_end)),
                context_positions=tuple(range(body_start, body_end)),
            )
        )
    return ScoutInput(
        tuple(ids),
        tuple(body_mask),
        tuple(prompt_mask),
        tuple(spans),
        common_prefix_end,
    )


def build_prompt_isolation_mask(
    layout: ScoutInput,
    *,
    scope: str,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Return an additive ``[1, 1, length, length]`` causal mask.

    Real text never reads semantic compression prompts. Each prompt reads its
    own chunk in ``local`` mode, or all text through its own chunk in
    ``cumulative`` mode. Different prompt slots remain mutually hidden.
    """
    if scope not in {"local", "cumulative"}:
        raise ValueError("scope must be local or cumulative")
    length = len(layout.input_ids)
    minimum = torch.finfo(dtype).min
    allowed = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    prompt_mask = torch.tensor(
        layout.compression_prompt_mask, device=device, dtype=torch.bool
    )
    body = torch.tensor(layout.body_mask, device=device, dtype=torch.bool)

    # Semantic compression prompts are invisible to all real-text rows.
    allowed[body, :] &= ~prompt_mask.unsqueeze(0)

    cumulative_body = torch.zeros(length, dtype=torch.bool, device=device)
    for span in layout.chunks:
        current = torch.zeros(length, dtype=torch.bool, device=device)
        candidate_positions = span.candidate_positions or tuple(
            range(span.body_start, span.body_end)
        )
        current[list(candidate_positions)] = True
        cumulative_body |= current
        # Isolate the full prompt, not only its final anchor token.
        prompt_start = span.prompt_start
        prompt_end = span.anchor_end
        visible_body = current if scope == "local" else cumulative_body
        for row in range(prompt_start, prompt_end):
            allowed[row, :] = False
            allowed[row, : layout.common_prefix_end] = True
            allowed[row, visible_body] = True
            if span.context_positions:
                allowed[row, list(span.context_positions)] = True
            allowed[row, prompt_start : row + 1] = True

    additive = torch.full((length, length), minimum, dtype=dtype, device=device)
    additive.masked_fill_(allowed, 0)
    return additive.unsqueeze(0).unsqueeze(0)


def continuous_position_ids(length: int, device: torch.device | str) -> torch.Tensor:
    return torch.arange(length, device=device, dtype=torch.long).unsqueeze(0)
