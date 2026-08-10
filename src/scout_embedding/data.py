"""Small, explicit retrieval dataset interface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

LONGEMBED_REPOSITORY = "dwzhu/LongEmbed"
LONGEMBED_REVISION = "6e346642246bfb4928c560ee08640dc84d074e8c"
LONGEMBED_SUBSETS = {
    "QMSum": "qmsum",
    "2WikiMultihopQA": "2wikimqa",
    "SummScreenFD": "summ_screen_fd",
    "NarrativeQA": "narrativeqa",
}


@dataclass(frozen=True)
class RetrievalDataset:
    queries: dict[str, str]
    corpus: dict[str, str]
    qrels: dict[str, dict[str, int]]


def load_longembed(name: str) -> RetrievalDataset:
    if name not in LONGEMBED_SUBSETS:
        raise KeyError(f"unknown LongEmbed task: {name}")
    dataset = load_dataset(
        LONGEMBED_REPOSITORY,
        name=LONGEMBED_SUBSETS[name],
        revision=LONGEMBED_REVISION,
    )
    qrels: dict[str, dict[str, int]] = {}
    for row in dataset["qrels"]:
        qrels.setdefault(str(row["qid"]), {})[str(row["doc_id"])] = 1
    queries = {
        str(row["qid"]): str(row["text"])
        for row in dataset["queries"]
        if str(row["qid"]) in qrels
    }
    corpus = {str(row["doc_id"]): str(row["text"]) for row in dataset["corpus"]}
    return RetrievalDataset(queries, corpus, qrels)


def load_jsonl_retrieval(path: str | Path) -> RetrievalDataset:
    root = Path(path)

    def rows(filename: str):
        with (root / filename).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    queries = {str(row["qid"]): str(row["text"]) for row in rows("queries.jsonl")}
    corpus = {str(row["doc_id"]): str(row["text"]) for row in rows("corpus.jsonl")}
    qrels: dict[str, dict[str, int]] = {}
    for row in rows("qrels.jsonl"):
        qrels.setdefault(str(row["qid"]), {})[str(row["doc_id"])] = int(row.get("score", 1))
    return RetrievalDataset(queries, corpus, qrels)
