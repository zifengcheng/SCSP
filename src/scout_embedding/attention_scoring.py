"""Semantic compression-guided attention scoring across decoder layers."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field

import torch

from .input_construction import SemanticChunkSpan


def scaled_dot_product_logits(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Return raw attention logits with shape ``[heads, prompts, tokens]``."""
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query and key must have shape [heads, positions, head_dim]")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    return query.float() @ key.float().transpose(-2, -1) / math.sqrt(query.shape[-1])


def _strongest_attention(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1).amax(dim=(0, 1))


def _l1_distribution(scores: torch.Tensor) -> torch.Tensor:
    return scores / scores.sum().clamp_min(torch.finfo(scores.dtype).eps)


def maximum_attention_distribution(layer_logits: list[torch.Tensor]) -> torch.Tensor:
    """Build SCOUT's normalized maximum-attention token distribution.

    Each input has shape ``[heads, prompts, tokens]``. Every attention row is
    softmax-normalized over the current chunk, then the strongest probability
    is retained for each token across prompts, heads, and layers. A final L1
    normalization makes the resulting vector a chunk-level distribution.
    """
    if not layer_logits:
        raise ValueError("layer_logits cannot be empty")
    token_count = layer_logits[0].shape[-1]
    if token_count == 0:
        raise ValueError("a score window must contain at least one token")
    layer_scores = []
    for logits in layer_logits:
        if logits.ndim != 3 or logits.shape[-1] != token_count:
            raise ValueError("all logits must have shape [heads, prompts, tokens]")
        layer_scores.append(_strongest_attention(logits))
    maximum = torch.stack(layer_scores).amax(dim=0)
    return _l1_distribution(maximum)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos.unsqueeze(0) + rotate_half(x) * sin.unsqueeze(0)


def rotary_cos_sin(rotary, sample: torch.Tensor, positions: torch.Tensor):
    parameters = inspect.signature(rotary.forward).parameters
    if "position_ids" in parameters or "seq_len" not in parameters:
        return rotary(sample, positions)
    sequence_length = int(positions.max().item()) + 1
    cos, sin = rotary(sample, seq_len=sequence_length)
    flat = positions.reshape(-1)
    return (
        cos.index_select(0, flat).reshape(*positions.shape, cos.shape[-1]),
        sin.index_select(0, flat).reshape(*positions.shape, sin.shape[-1]),
    )


def decoder_backbone(model: torch.nn.Module) -> torch.nn.Module:
    candidates = [model, getattr(model, "model", None), getattr(model, "base_model", None)]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise TypeError(f"cannot locate decoder layers on {type(model).__name__}")


@dataclass
class SemanticCompressionAttentionCollector:
    """Capture dense keys and sparse compression-prompt queries per layer."""

    model: torch.nn.Module
    chunks: tuple[SemanticChunkSpan, ...] = ()
    position_ids: torch.Tensor | None = None
    _queries: dict[int, list[torch.Tensor]] = field(default_factory=dict, init=False)
    _keys: dict[int, torch.Tensor] = field(default_factory=dict, init=False)
    _handles: list = field(default_factory=list, init=False)

    def install(self) -> None:
        if self._handles:
            raise RuntimeError("collector is already installed")
        for index, layer in enumerate(decoder_backbone(self.model).layers):
            self._handles.append(
                layer.self_attn.q_proj.register_forward_hook(self._query_hook(index))
            )
            self._handles.append(
                layer.self_attn.k_proj.register_forward_hook(self._key_hook(index))
            )

    def _query_hook(self, layer: int):
        def hook(_module, _inputs, output):
            self._queries[layer] = [
                output[0, span.anchor_start : span.anchor_end].detach()
                for span in self.chunks
            ]

        return hook

    def _key_hook(self, layer: int):
        def hook(_module, _inputs, output):
            self._keys[layer] = output.detach()

        return hook

    @torch.no_grad()
    def token_scores(self) -> list[torch.Tensor]:
        if not self._keys or self.position_ids is None:
            raise RuntimeError("run a model forward after configuring the collector")
        backbone = decoder_backbone(self.model)
        config = backbone.config
        q_heads = int(config.num_attention_heads)
        kv_heads = int(getattr(config, "num_key_value_heads", q_heads))
        head_dim = int(getattr(config, "head_dim", config.hidden_size // q_heads))
        groups = q_heads // kv_heads
        positions = self.position_ids
        rotary = getattr(backbone, "rotary_emb", None)
        if rotary is None:
            rotary = backbone.layers[0].self_attn.rotary_emb
        sample = next(iter(self._keys.values()))[..., :head_dim]
        cos, sin = rotary_cos_sin(rotary, sample, positions)
        cos, sin = cos[0], sin[0]

        raw_max = [
            torch.zeros(
                len(span.candidate_positions)
                if span.candidate_positions
                else span.body_end - span.body_start,
                device=self._keys[0].device,
            )
            for span in self.chunks
        ]
        for layer_index in range(len(backbone.layers)):
            key_raw = self._keys[layer_index][0]
            for chunk_index, span in enumerate(self.chunks):
                query = self._queries[layer_index][chunk_index]
                query = query.view(-1, q_heads, head_dim).transpose(0, 1)
                key_positions = torch.tensor(
                    span.candidate_positions
                    or tuple(range(span.body_start, span.body_end)),
                    device=query.device,
                    dtype=torch.long,
                )
                key = key_raw.index_select(0, key_positions)
                key = key.view(-1, kv_heads, head_dim).transpose(0, 1)
                query_positions = torch.arange(
                    span.anchor_start, span.anchor_end, device=query.device
                )
                query = apply_rope(query, cos[query_positions], sin[query_positions])
                key = apply_rope(key, cos[key_positions], sin[key_positions])
                key = key.index_select(
                    0, torch.arange(q_heads, device=key.device) // groups
                )
                layer_max = _strongest_attention(scaled_dot_product_logits(query, key))
                raw_max[chunk_index] = torch.maximum(raw_max[chunk_index], layer_max)
        return [_l1_distribution(score) for score in raw_max]

    def clear(self) -> None:
        self._queries.clear()
        self._keys.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *_exc):
        self.remove()
