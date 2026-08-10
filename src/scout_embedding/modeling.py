"""Unified long-context encoder for SCOUT and its parent methods."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .attention_scoring import SemanticCompressionAttentionCollector, decoder_backbone
from .baselines.echo import build_echo_input
from .baselines.htp import HTPRewirer, build_htp_layout, ensure_htp_tokens
from .chunking import (
    TokenChunk,
    fixed_token_chunks,
    sentence_budget_chunks,
    sentence_token_units,
    truncate_chunks,
)
from .config import ExperimentConfig
from .input_construction import (
    ScoutInput,
    SemanticChunkSpan,
    build_prompt_isolation_mask,
    build_scout_input,
    continuous_position_ids,
)
from .selective_pooling import normalized_masked_mean
from .token_selection import select_above_uniform_attention


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class HiddenCapture:
    def __init__(self, model, layer_index: int):
        layers = decoder_backbone(model).layers
        resolved = len(layers) + layer_index if layer_index < 0 else layer_index
        if not 0 <= resolved < len(layers):
            raise ValueError(f"readout layer {layer_index} is outside {len(layers)} layers")
        self.hidden: torch.Tensor | None = None
        self.handle = layers[resolved].register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.hidden = (output[0] if isinstance(output, tuple) else output).detach()

    def clear(self) -> None:
        self.hidden = None

    def remove(self) -> None:
        self.handle.remove()


@dataclass
class EncoderStats:
    documents: int = 0
    body_tokens: int = 0
    physical_tokens: int = 0
    retained_tokens: int = 0
    chunks: int = 0

    def as_dict(self) -> dict[str, float]:
        denominator = max(self.documents, 1)
        return {
            "documents": self.documents,
            "body_tokens_avg": self.body_tokens / denominator,
            "physical_tokens_avg": self.physical_tokens / denominator,
            "retained_fraction": self.retained_tokens / max(self.body_tokens, 1),
            "chunks_per_document": self.chunks / denominator,
        }


class LongContextEmbeddingEncoder:
    """One decoder instance shared across every dataset in an experiment."""

    def __init__(self, config: ExperimentConfig, device: str | None = None):
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = resolve_dtype(config.model.dtype, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.name_or_path,
            revision=config.model.revision,
            trust_remote_code=config.model.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModel.from_pretrained(
                config.model.name_or_path,
                revision=config.model.revision,
                torch_dtype=self.dtype,
                attn_implementation=config.model.attention_backend,
                trust_remote_code=config.model.trust_remote_code,
            )
            .to(self.device)
            .eval()
        )
        self.capture = HiddenCapture(self.model, config.model.readout_layer)
        self.stats = EncoderStats()
        self._htp_tokens: tuple[int, int] | None = None

    def close(self) -> None:
        self.capture.remove()

    def reset_stats(self) -> None:
        self.stats = EncoderStats()

    def _body_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )[: self.config.model.max_body_tokens]

    def _semantic_chunks(self, text: str) -> list[TokenChunk]:
        scout = self.config.scout
        if scout.chunking == "fixed":
            chunks = fixed_token_chunks(self._body_ids(text), scout.chunk_size)
        else:
            chunks = sentence_budget_chunks(
                text,
                self.tokenizer,
                scout.chunk_size,
                model_name=scout.sentence_splitter,
            )
        return truncate_chunks(chunks, self.config.model.max_body_tokens)

    def _forward(self, layout: ScoutInput, custom_mask: bool = False) -> torch.Tensor:
        position_limit = int(
            getattr(decoder_backbone(self.model).config, "max_position_embeddings", 0) or 0
        )
        if position_limit and len(layout.input_ids) > position_limit:
            raise ValueError(
                f"physical sequence length {len(layout.input_ids)} exceeds the "
                f"model position limit {position_limit}"
            )
        ids = torch.tensor([layout.input_ids], device=self.device, dtype=torch.long)
        positions = continuous_position_ids(ids.shape[1], self.device)
        if custom_mask:
            attention = build_prompt_isolation_mask(
                layout,
                scope=self.config.scout.prompt_attention_scope,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            attention = torch.ones_like(ids)
        self.capture.clear()
        self.model(
            input_ids=ids,
            attention_mask=attention,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        )
        if self.capture.hidden is None:
            raise RuntimeError("readout hook did not capture hidden states")
        return self.capture.hidden

    def _forward_batch(self, layouts: list[ScoutInput]) -> torch.Tensor:
        maximum = max(len(layout.input_ids) for layout in layouts)
        position_limit = int(
            getattr(decoder_backbone(self.model).config, "max_position_embeddings", 0) or 0
        )
        if position_limit and maximum > position_limit:
            raise ValueError(
                f"physical sequence length {maximum} exceeds the model position "
                f"limit {position_limit}"
            )
        pad = int(self.tokenizer.pad_token_id)
        ids = torch.full((len(layouts), maximum), pad, device=self.device, dtype=torch.long)
        attention = torch.zeros_like(ids)
        for row, layout in enumerate(layouts):
            length = len(layout.input_ids)
            ids[row, :length] = torch.tensor(layout.input_ids, device=self.device)
            attention[row, :length] = 1
        positions = torch.arange(maximum, device=self.device).unsqueeze(0).expand_as(ids)
        self.capture.clear()
        self.model(
            input_ids=ids,
            attention_mask=attention,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        )
        if self.capture.hidden is None:
            raise RuntimeError("readout hook did not capture hidden states")
        return self.capture.hidden

    def _plain_layout(self, text: str, instruction: str = "") -> ScoutInput:
        ids: list[int] = []
        mask: list[bool] = []
        bos = self.tokenizer.bos_token_id
        if bos is not None:
            ids.append(bos)
            mask.append(False)
        if instruction:
            prefix = self.tokenizer.encode(
                f"{instruction} Text: ", add_special_tokens=False
            )
            ids.extend(prefix)
            mask.extend([False] * len(prefix))
        body = self._body_ids(text)
        ids.extend(body)
        mask.extend([True] * len(body))
        return ScoutInput(
            tuple(ids), tuple(mask), tuple(False for _ in ids), (), len(ids) - len(body)
        )

    @torch.inference_mode()
    def encode_mean(
        self, texts: Iterable[str], instruction: str = "", batch_size: int = 1
    ) -> np.ndarray:
        texts = list(texts)
        embeddings = []
        for start in range(0, len(texts), batch_size):
            layouts = [
                self._plain_layout(text, instruction)
                for text in texts[start : start + batch_size]
            ]
            hidden = self._forward_batch(layouts)
            mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=self.device)
            for row, layout in enumerate(layouts):
                mask[row, : len(layout.body_mask)] = torch.tensor(
                    layout.body_mask, device=self.device
                )
            pooled = normalized_masked_mean(hidden, mask).cpu()
            embeddings.extend(pooled)
            for layout in layouts:
                self._record(layout, sum(layout.body_mask))
        return torch.stack(embeddings).numpy()

    def _echo_layout(self, text: str, instruction: str = "") -> ScoutInput:
        body = self._body_ids(text)
        if instruction:
            prefix = self.tokenizer.encode(
                f"{instruction} Query: ", add_special_tokens=False
            )
            separator = self.tokenizer.encode(" Query again: ", add_special_tokens=False)
        else:
            prefix, separator = [], []
        return build_echo_input(
            body,
            bos_token_id=self.tokenizer.bos_token_id,
            instruction_token_ids=prefix,
            repetition_separator_ids=separator,
        )

    @torch.inference_mode()
    def encode_echo(
        self, texts: Iterable[str], instruction: str = "", batch_size: int = 1
    ) -> np.ndarray:
        texts = list(texts)
        embeddings = []
        for start in range(0, len(texts), batch_size):
            layouts = [
                self._echo_layout(text, instruction)
                for text in texts[start : start + batch_size]
            ]
            hidden = self._forward_batch(layouts)
            mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=self.device)
            for row, layout in enumerate(layouts):
                mask[row, : len(layout.body_mask)] = torch.tensor(
                    layout.body_mask, device=self.device
                )
            embeddings.extend(normalized_masked_mean(hidden, mask).cpu())
            for layout in layouts:
                self._record(layout, sum(layout.body_mask))
        return torch.stack(embeddings).numpy()

    def _scout_input(self, text: str, parent: str) -> ScoutInput:
        chunks = self._semantic_chunks(text)
        scout = self.config.scout
        prompt = self.tokenizer.encode(scout.compression_prompt, add_special_tokens=False)
        if parent == "vanilla":
            return build_scout_input(
                chunks,
                bos_token_id=self.tokenizer.bos_token_id,
                compression_prompt_token_ids=prompt,
                anchor_tokens=scout.prompt_anchor_tokens,
            )
        if parent == "echo":
            # Preserve the complete Echo input, then append SCOUT prompts.
            body = [token for chunk in chunks for token in chunk.token_ids]
            echo_input = build_echo_input(body, bos_token_id=self.tokenizer.bos_token_id)
            ids = list(echo_input.input_ids)
            body_mask = list(echo_input.body_mask)
            prompt_mask = list(echo_input.compression_prompt_mask)
            second_start = body_mask.index(True)
            candidates_per_chunk: list[list[int]] = []
            cursor = second_start
            for chunk in chunks:
                positions = list(range(cursor, cursor + len(chunk.token_ids)))
                candidates_per_chunk.append(positions)
                cursor += len(chunk.token_ids)
            spans = []
            slot_ids = prompt
            for candidates in candidates_per_chunk:
                prompt_start = len(ids)
                ids.extend(slot_ids)
                body_mask.extend([False] * len(slot_ids))
                prompt_mask.extend([True] * len(slot_ids))
                prompt_end = len(ids)
                spans.append(
                    SemanticChunkSpan(
                        body_start=candidates[0],
                        body_end=candidates[-1] + 1,
                        anchor_start=prompt_end - scout.prompt_anchor_tokens,
                        anchor_end=prompt_end,
                        prompt_start=prompt_start,
                        candidate_positions=tuple(candidates),
                        context_positions=tuple(candidates),
                    )
                )
            return ScoutInput(
                tuple(ids),
                tuple(body_mask),
                tuple(prompt_mask),
                tuple(spans),
                second_start,
            )
        raise ValueError(f"unknown SCOUT parent: {parent}")

    @torch.inference_mode()
    def encode_scout(self, texts: Iterable[str], parent: str = "vanilla") -> np.ndarray:
        embeddings = []
        for text in texts:
            layout = self._scout_input(text, parent)
            positions = continuous_position_ids(len(layout.input_ids), self.device)
            collector = SemanticCompressionAttentionCollector(
                self.model, chunks=layout.chunks, position_ids=positions
            )
            with collector:
                hidden = self._forward(layout, custom_mask=True)
                token_scores = collector.token_scores()
            selection_mask = torch.zeros(
                (1, len(layout.input_ids)), dtype=torch.bool, device=self.device
            )
            for span, probabilities in zip(layout.chunks, token_scores):
                local = select_above_uniform_attention(
                    probabilities,
                    self.config.scout.selection_ratio,
                    min_keep=self.config.scout.min_tokens_per_chunk,
                )
                candidates = torch.tensor(
                    span.candidate_positions, device=self.device, dtype=torch.long
                )
                selection_mask[0, candidates[local]] = True
            embeddings.append(normalized_masked_mean(hidden, selection_mask)[0].cpu())
            self._record(layout, int(selection_mask.sum()))
        return torch.stack(embeddings).numpy()

    def _htp_layout(self, text: str, use_scout: bool, instruction: str = ""):
        htp = self.config.htp
        units = sentence_token_units(
            text, self.tokenizer, model_name=self.config.scout.sentence_splitter
        )
        blocks = [
            [
                token
                for unit in units[start : start + htp.sentences_per_block]
                for token in unit
            ]
            for start in range(0, len(units), htp.sentences_per_block)
        ]
        if self._htp_tokens is None:
            self._htp_tokens = ensure_htp_tokens(self.tokenizer, self.model)
        pst, bpst = self._htp_tokens
        compression_prompt = (
            self.tokenizer.encode(
                self.config.scout.compression_prompt, add_special_tokens=False
            )
            if use_scout
            else None
        )
        leading = None
        if instruction:
            leading = []
            if self.tokenizer.bos_token_id is not None:
                leading.append(self.tokenizer.bos_token_id)
            leading.extend(
                self.tokenizer.encode(f"{instruction} Text: ", add_special_tokens=False)
            )
        return build_htp_layout(
            blocks,
            bos_token_id=self.tokenizer.bos_token_id,
            pst_token_id=pst,
            bpst_token_id=bpst,
            max_body_tokens=self.config.model.max_body_tokens,
            scout_chunk_size=self.config.scout.chunk_size if use_scout else None,
            compression_prompt_token_ids=compression_prompt,
            anchor_tokens=self.config.scout.prompt_anchor_tokens,
            leading_token_ids=leading,
        )

    @torch.inference_mode()
    def encode_htp(
        self,
        texts: Iterable[str],
        use_scout: bool = False,
        instruction: str = "",
    ) -> np.ndarray:
        embeddings = []
        htp_config = self.config.htp
        with HTPRewirer(
            self.model, htp_config.rewiring_start, htp_config.rewiring_end
        ) as rewirer:
            for text in texts:
                htp_layout = self._htp_layout(text, use_scout, instruction)
                rewirer.set_layout(htp_layout)
                layout = htp_layout.model_input
                if use_scout:
                    positions = continuous_position_ids(len(layout.input_ids), self.device)
                    collector = SemanticCompressionAttentionCollector(
                        self.model, chunks=layout.chunks, position_ids=positions
                    )
                    with collector:
                        hidden = self._forward(layout, custom_mask=True)
                        token_scores = collector.token_scores()
                    pool_mask = torch.zeros(
                        (1, len(layout.input_ids)), dtype=torch.bool, device=self.device
                    )
                    for span, probabilities in zip(layout.chunks, token_scores):
                        local = select_above_uniform_attention(
                            probabilities,
                            self.config.scout.selection_ratio,
                            min_keep=self.config.scout.min_tokens_per_chunk,
                        )
                        candidates = torch.tensor(
                            span.candidate_positions, device=self.device
                        )
                        pool_mask[0, candidates[local]] = True
                else:
                    hidden = self._forward(layout)
                    pool_mask = torch.tensor([layout.body_mask], device=self.device)
                embeddings.append(normalized_masked_mean(hidden, pool_mask)[0].cpu())
                self._record(layout, int(pool_mask.sum()))
        return torch.stack(embeddings).numpy()

    def encode(
        self, texts: Iterable[str], *, is_query: bool, instruction: str = ""
    ) -> np.ndarray:
        texts = list(texts)
        method = self.config.method
        batch_size = (
            self.config.evaluation.query_batch_size
            if is_query
            else self.config.evaluation.batch_size
        )
        if is_query:
            if method in {"echo", "echo_scout"}:
                return self.encode_echo(texts, instruction, batch_size=batch_size)
            if method in {"htp", "htp_scout"}:
                return self.encode_htp(texts, use_scout=False, instruction=instruction)
            return self.encode_mean(texts, instruction, batch_size=batch_size)
        if method == "vanilla":
            return self.encode_mean(texts, batch_size=batch_size)
        if method == "scout":
            return self.encode_scout(texts)
        if method == "echo":
            return self.encode_echo(texts, batch_size=batch_size)
        if method == "echo_scout":
            return self.encode_scout(texts, parent="echo")
        if method == "htp":
            return self.encode_htp(texts)
        if method == "htp_scout":
            return self.encode_htp(texts, use_scout=True)
        raise AssertionError(method)

    def _record(self, layout: ScoutInput, retained: int) -> None:
        self.stats.documents += 1
        self.stats.body_tokens += sum(layout.body_mask)
        self.stats.physical_tokens += len(layout.input_ids)
        self.stats.retained_tokens += retained
        self.stats.chunks += len(layout.chunks)
