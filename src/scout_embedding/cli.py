"""Command-line evaluation entry point."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .config import ExperimentConfig
from .data import load_jsonl_retrieval, load_longembed
from .evaluation import evaluate_ndcg
from .modeling import LongContextEmbeddingEncoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--device", default=None, help="Optional Torch device override")
    parser.add_argument(
        "--local-dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add a BEIR-style queries/corpus/qrels JSONL directory",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    local = dict(item.split("=", 1) for item in args.local_dataset)
    output = Path(config.evaluation.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    encoder = LongContextEmbeddingEncoder(config, device=args.device)
    try:
        for name in config.evaluation.datasets:
            dataset = (
                load_jsonl_retrieval(local[name]) if name in local else load_longembed(name)
            )
            query_ids = list(dataset.queries)
            document_ids = list(dataset.corpus)
            start = time.perf_counter()
            query_embeddings = encoder.encode(
                dataset.queries.values(),
                is_query=True,
                instruction=config.evaluation.query_instruction,
            )
            encoder.reset_stats()
            document_embeddings = encoder.encode(dataset.corpus.values(), is_query=False)
            metrics = evaluate_ndcg(
                query_embeddings,
                document_embeddings,
                query_ids,
                document_ids,
                dataset.qrels,
            )
            row = {
                "dataset": name,
                "method": config.method,
                **metrics,
                **encoder.stats.as_dict(),
                "wall_seconds": time.perf_counter() - start,
            }
            rows.append(row)
            (output / f"{name}.json").write_text(
                json.dumps({"config": config.to_dict(), "metrics": row}, indent=2),
                encoding="utf-8",
            )
    finally:
        encoder.close()
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
