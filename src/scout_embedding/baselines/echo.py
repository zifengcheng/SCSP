"""Echo input construction shared by the baseline and SCOUT composition."""

from __future__ import annotations

from ..input_construction import ScoutInput


def build_echo_input(
    body_token_ids: list[int],
    *,
    bos_token_id: int | None,
    instruction_token_ids: list[int] | None = None,
    repetition_separator_ids: list[int] | None = None,
) -> ScoutInput:
    """Repeat a sequence and mark only its second occurrence for pooling."""
    ids: list[int] = []
    body_mask: list[bool] = []
    if bos_token_id is not None:
        ids.append(bos_token_id)
        body_mask.append(False)

    instruction = instruction_token_ids or []
    separator = repetition_separator_ids or []
    ids.extend(instruction)
    body_mask.extend([False] * len(instruction))
    ids.extend(body_token_ids)
    body_mask.extend([False] * len(body_token_ids))
    ids.extend(separator)
    body_mask.extend([False] * len(separator))
    ids.extend(body_token_ids)
    body_mask.extend([True] * len(body_token_ids))

    return ScoutInput(
        input_ids=tuple(ids),
        body_mask=tuple(body_mask),
        compression_prompt_mask=tuple(False for _ in ids),
        chunks=(),
        common_prefix_end=0,
    )
