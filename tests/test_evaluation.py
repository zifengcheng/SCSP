import numpy as np

from scout_embedding.evaluation import evaluate_ndcg


def test_perfect_ranking_has_unit_ndcg():
    queries = np.eye(2, dtype=np.float32)
    documents = np.eye(2, dtype=np.float32)
    metrics = evaluate_ndcg(
        queries,
        documents,
        ["q1", "q2"],
        ["d1", "d2"],
        {"q1": {"d1": 1}, "q2": {"d2": 1}},
    )
    assert metrics == {"ndcg@1": 1.0, "ndcg@10": 1.0}
