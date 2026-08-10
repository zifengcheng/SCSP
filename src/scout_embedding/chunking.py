"""Document chunking policies used by SCOUT and the HTP baseline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class TokenChunk:
    token_ids: tuple[int, ...]
    sentence_count: int
    oversize: bool = False


@lru_cache(maxsize=4)
def load_spacy_pipeline(model_name: str):
    try:
        import spacy

        return spacy.load(
            model_name,
            disable=["ner", "tagger", "lemmatizer", "attribute_ruler"],
        )
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {model_name!r} is not installed. Run: "
            f"python -m spacy download {model_name}"
        ) from exc


def fixed_token_chunks(token_ids: list[int], budget: int) -> list[TokenChunk]:
    if budget < 1:
        raise ValueError("budget must be positive")
    return [
        TokenChunk(tuple(token_ids[start : start + budget]), sentence_count=0)
        for start in range(0, len(token_ids), budget)
    ]


def sentence_budget_chunks(
    text: str,
    tokenizer: Tokenizer,
    budget: int,
    *,
    model_name: str = "en_core_web_sm",
) -> list[TokenChunk]:
    """Pack complete spaCy sentences up to a soft token budget.

    An individual sentence longer than the budget remains intact and is marked
    as oversize. It is never silently split into fixed-token fragments.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    nlp = load_spacy_pipeline(model_name)
    doc = nlp(text)
    sentence_ids = [
        tokenizer.encode(sent.text_with_ws, add_special_tokens=False)
        for sent in doc.sents
        if sent.text_with_ws
    ]
    if not sentence_ids and text:
        sentence_ids = [tokenizer.encode(text, add_special_tokens=False)]

    chunks: list[TokenChunk] = []
    current: list[int] = []
    sentence_count = 0
    for ids in sentence_ids:
        if current and len(current) + len(ids) > budget:
            chunks.append(TokenChunk(tuple(current), sentence_count))
            current = []
            sentence_count = 0
        if len(ids) > budget:
            if current:
                chunks.append(TokenChunk(tuple(current), sentence_count))
                current = []
                sentence_count = 0
            chunks.append(TokenChunk(tuple(ids), 1, oversize=True))
            continue
        current.extend(ids)
        sentence_count += 1
    if current:
        chunks.append(TokenChunk(tuple(current), sentence_count))
    return chunks


def sentence_token_units(
    text: str,
    tokenizer: Tokenizer,
    *,
    model_name: str = "en_core_web_sm",
) -> list[list[int]]:
    """Tokenize the same spaCy sentence units used by HTP and semantic chunks."""
    nlp = load_spacy_pipeline(model_name)
    units = [
        tokenizer.encode(sent.text_with_ws, add_special_tokens=False)
        for sent in nlp(text).sents
        if sent.text_with_ws
    ]
    return units or ([tokenizer.encode(text, add_special_tokens=False)] if text else [])


def truncate_chunks(chunks: list[TokenChunk], max_tokens: int) -> list[TokenChunk]:
    """Retain at most ``max_tokens`` body tokens, truncating the final chunk."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    retained: list[TokenChunk] = []
    used = 0
    for chunk in chunks:
        remaining = max_tokens - used
        if remaining <= 0:
            break
        ids = chunk.token_ids[:remaining]
        if ids:
            retained.append(TokenChunk(ids, chunk.sentence_count, chunk.oversize))
            used += len(ids)
    return retained
