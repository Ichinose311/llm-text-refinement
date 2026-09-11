import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def load_rows(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def key_of(row):
    return (
        row["source_file"],
        int(row["data_id"]),
        int(row["candidate_index"]),
    )


def minmax(values: List[float]) -> List[float]:
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def dcg(labels: List[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(labels[:k]))


def metrics_for_group(labels: List[int], ks: List[int]) -> Dict[str, float]:
    result = {}
    positives = sum(1 for x in labels if x > 0)

    for k in ks:
        topk = labels[:k]
        hits = sum(1 for x in topk if x > 0)

        result[f"Recall@{k}"] = hits / positives if positives else 0.0

        ideal = sorted(labels, reverse=True)
        ideal_dcg = dcg(ideal, k)
        result[f"NDCG@{k}"] = dcg(labels, k) / ideal_dcg if ideal_dcg > 0 else 0.0

        mrr = 0.0
        for rank, rel in enumerate(topk, start=1):
            if rel > 0:
                mrr = 1.0 / rank
                break
        result[f"MRR@{k}"] = mrr

    return result


def average(metrics):
    keys = sorted(metrics[0].keys())
    return {k: float(np.mean([m[k] for m in metrics])) for k in keys}


def evaluate(rows, alpha, ks):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source_file"], int(row["data_id"]))].append(row)

    metrics = []

    for _, group in grouped.items():
        baseline_scores = [float(x["baseline_score"]) for x in group]
        reason_scores = [float(x["reason_score"]) for x in group]

        baseline_norm = minmax(baseline_scores)
        reason_norm = minmax(reason_scores)

        fused = []
        for row, b, r in zip(group, baseline_norm, reason_norm):
            new_row = dict(row)
            new_row["fusion_score"] = alpha * b + (1.0 - alpha) * r
            fused.append(new_row)

        ranked = sorted(fused, key=lambda x: x["fusion_score"], reverse=True)
        labels = [int(x["score"]) for x in ranked]
        metrics.append(metrics_for_group(labels, ks))

    return average(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-scored", required=True)
    parser.add_argument("--reason-scored", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    baseline_rows = load_rows(args.baseline_scored)
    reason_rows = load_rows(args.reason_scored)

    baseline_map = {key_of(x): x for x in baseline_rows}
    reason_map = {key_of(x): x for x in reason_rows}

    keys = sorted(set(baseline_map) & set(reason_map))

    rows = []
    for key in keys:
        b = baseline_map[key]
        r = reason_map[key]

        rows.append({
            "source_file": b["source_file"],
            "data_id": int(b["data_id"]),
            "candidate_index": int(b["candidate_index"]),
            "score": int(b["score"]),
            "baseline_score": float(b["predicted_score"]),
            "reason_score": float(r["predicted_score"]),
            "recommendation_sentence": r.get("recommendation_sentence", ""),
        })

    print("aligned rows:", len(rows))
    print("groups:", len(set((x["source_file"], x["data_id"]) for x in rows)))

    alphas = []
    x = 0.0
    while x <= 1.000001:
        alphas.append(round(x, 6))
        x += args.step

    results = []
    for alpha in alphas:
        avg = evaluate(rows, alpha, args.ks)
        avg["alpha_baseline"] = alpha
        results.append(avg)

    target = "NDCG@5" if "NDCG@5" in results[0] else sorted(results[0].keys())[0]
    best = max(results, key=lambda x: x[target])

    print("\n=== Fusion Grid ===")
    for r in results:
        print(
            f"alpha={r['alpha_baseline']:.2f} "
            f"Recall@5={r.get('Recall@5', 0):.6f} "
            f"NDCG@5={r.get('NDCG@5', 0):.6f} "
            f"MRR@5={r.get('MRR@5', 0):.6f}"
        )

    print("\n=== Best ===")
    for k in sorted(best.keys()):
        if k == "alpha_baseline":
            print(f"{k}: {best[k]:.2f}")
        else:
            print(f"{k}: {best[k]:.6f}")


if __name__ == "__main__":
    main()
