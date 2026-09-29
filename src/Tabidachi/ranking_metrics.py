"""CPU-only ranking metrics, independent of model loading.

Groups are (source_file, data_id). Ties preserve input order. Relevance is a
nonnegative integer (research data uses 0/1). NDCG uses linear gain, matching the
original precomputed evaluator. Groups without positives contribute zero.
"""

import math
from collections import defaultdict
from statistics import mean


def validate_ks(ks):
    ks = list(ks)
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    return ks


def dcg(labels, k):
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(labels[:k]))


def metrics_for_group(labels, ks):
    """Evaluate already ranked labels; HR and Recall differ with >1 positive."""
    ks = validate_ks(ks)
    labels = list(labels)
    if not labels:
        raise ValueError("A ranking group must contain at least one candidate")
    if any(type(v) is not int or v < 0 for v in labels):
        raise ValueError("Relevance labels must be nonnegative integers")
    positives = sum(v > 0 for v in labels)
    ideal = sorted(labels, reverse=True)
    result = {}
    for k in ks:
        topk = labels[:k]
        hits = sum(v > 0 for v in topk)
        ideal_dcg = dcg(ideal, k)
        result[f"HR@{k}"] = float(hits > 0)
        result[f"Recall@{k}"] = hits / positives if positives else 0.0
        result[f"MRR@{k}"] = next(
            (1.0 / rank for rank, rel in enumerate(topk, 1) if rel > 0), 0.0
        )
        result[f"NDCG@{k}"] = dcg(labels, k) / ideal_dcg if ideal_dcg else 0.0
    return result


def ranking_metrics(rows, ks, score_key="predicted_score"):
    """Return macro-averaged metrics and group count; reject invalid input."""
    ks = validate_ks(ks)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        try:
            source = row["source_file"]
            data_id = row["data_id"]
            label = row["score"]
            score = float(row[score_key])
            if not isinstance(source, str) or not source:
                raise ValueError("source_file must be a nonempty string")
            if type(data_id) is not int:
                raise ValueError("data_id must be an integer")
            if type(label) is not int or label < 0:
                raise ValueError("score must be a nonnegative integer")
            if not math.isfinite(score):
                raise ValueError(f"{score_key} must be finite")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid candidate row {index}: {exc}") from exc
        groups[(source, data_id)].append((score, label))
    if not groups:
        raise ValueError("No candidate rows found; check the input directory and JSON files")
    metrics = [
        metrics_for_group([label for _, label in sorted(items, key=lambda x: x[0], reverse=True)], ks)
        for items in groups.values()
    ]
    return {key: mean(m[key] for m in metrics) for key in metrics[0]}, len(groups)
