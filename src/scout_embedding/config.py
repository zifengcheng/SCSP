"""Typed experiment configuration with strict validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.3"
    revision: str | None = None
    max_body_tokens: int = 8192
    readout_layer: int = -3
    dtype: str = "auto"
    attention_backend: str = "sdpa"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if self.max_body_tokens < 1:
            raise ValueError("max_body_tokens must be positive")
        if self.readout_layer >= 0:
            raise ValueError("readout_layer must be a negative layer index")
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")


@dataclass(frozen=True)
class ScoutConfig:
    """SCOUT input construction, scoring, and selection parameters."""

    chunk_size: int = 512
    chunking: str = "sentence_aware"
    sentence_splitter: str = "en_core_web_sm"
    compression_prompt: str = '\nBrief summary:"'
    prompt_anchor_tokens: int = 1
    prompt_attention_scope: str = "local"
    position_mode: str = "continuous"
    attention_aggregation: str = "maximum"
    selection_ratio: float = 1.75
    min_tokens_per_chunk: int = 1

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.chunking not in {"sentence_aware", "fixed"}:
            raise ValueError("chunking must be sentence_aware or fixed")
        if self.prompt_attention_scope not in {"local", "cumulative"}:
            raise ValueError("prompt_attention_scope must be local or cumulative")
        if self.position_mode != "continuous":
            raise ValueError("SCOUT uses continuous position IDs")
        if self.attention_aggregation != "maximum":
            raise ValueError("SCOUT uses maximum aggregation across layers and heads")
        if self.selection_ratio < 0:
            raise ValueError("selection_ratio must be non-negative")
        if self.prompt_anchor_tokens < 1 or self.min_tokens_per_chunk < 1:
            raise ValueError("anchor and minimum keep counts must be positive")
        if not self.compression_prompt:
            raise ValueError("compression_prompt cannot be empty")


@dataclass(frozen=True)
class HTPConfig:
    sentences_per_block: int = 1
    rewiring_start: int = 1
    rewiring_end: int = 7

    def __post_init__(self) -> None:
        if self.sentences_per_block < 1:
            raise ValueError("sentences_per_block must be positive")
        if not 0 <= self.rewiring_start < self.rewiring_end:
            raise ValueError("HTP rewiring range must be non-empty")


@dataclass(frozen=True)
class EvaluationConfig:
    datasets: tuple[str, ...] = (
        "QMSum",
        "2WikiMultihopQA",
        "SummScreenFD",
        "NarrativeQA",
    )
    query_instruction: str = "Retrieve the relevant document"
    batch_size: int = 1
    query_batch_size: int = 16
    output_dir: str = "results/scout"

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("at least one evaluation dataset is required")
        if self.batch_size < 1 or self.query_batch_size < 1:
            raise ValueError("batch sizes must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    method: str = "scout"
    model: ModelConfig = field(default_factory=ModelConfig)
    scout: ScoutConfig = field(default_factory=ScoutConfig)
    htp: HTPConfig = field(default_factory=HTPConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        valid = {"vanilla", "scout", "echo", "echo_scout", "htp", "htp_scout"}
        if self.method not in valid:
            raise ValueError(f"method must be one of {sorted(valid)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            method=raw.get("method", "scout"),
            model=ModelConfig(**raw.get("model", {})),
            scout=ScoutConfig(**raw.get("scout", {})),
            htp=HTPConfig(**raw.get("htp", {})),
            evaluation=EvaluationConfig(
                **{
                    **raw.get("evaluation", {}),
                    "datasets": tuple(
                        raw.get("evaluation", {}).get(
                            "datasets", EvaluationConfig().datasets
                        )
                    ),
                }
            ),
        )
