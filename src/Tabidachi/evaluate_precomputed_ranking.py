import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


def load_rows(input_dir: str) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.load(f))
    return rows


def dcg(labels: List[int], k: int) -> float:
    value = 0.0
    for i, rel in enumerate(labels[:k]):
        value += rel / math.log2(i + 2)
    return value


def metrics_for_group(labels: List[int], ks: List[int]) -> Dict[str, float]:
    result = {}
    positives = sum(1 for x in labels if int(x) > 0)

    for k in ks:
        topk = labels[:k]
        hit_count = sum(1 for x in topk if int(x) > 0)

        result[f"Recall@{k}"] = hit_count / positives if positives > 0 else 0.0

        ideal = sorted(labels, reverse=True)
        ideal_dcg = dcg(ideal, k)
        result[f"NDCG@{k}"] = dcg(labels, k) / ideal_dcg if ideal_dcg > 0 else 0.0

        mrr = 0.0
        for rank, rel in enumerate(topk, start=1):
            if int(rel) > 0:
                mrr = 1.0 / rank
                break
        result[f"MRR@{k}"] = mrr

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--score-key", default="predicted_score")
    parser.add_argument("--save-scored", default=None)
    args = parser.parse_args()

    rows = load_rows(args.input_dir)
    print("loaded rows:", len(rows))

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source_file"], int(row["data_id"]))].append(row)

    print("groups:", len(grouped))

    metrics = []
    for _, group in grouped.items():
        ranked = sorted(group, key=lambda x: float(x[args.score_key]), reverse=True)
        labels = [int(x["score"]) for x in ranked]
        metrics.append(metrics_for_group(labels, args.ks))

    keys = sorted(metrics[0].keys())
    print("\n=== Ranking Metrics ===")
    for key in keys:
        print(f"{key}: {float(np.mean([m[key] for m in metrics])):.6f}")

    if args.save_scored:
        os.makedirs(os.path.dirname(args.save_scored), exist_ok=True)
        with open(args.save_scored, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=4)
        print(f"\nsaved scored rows to: {args.save_scored}")


if __name__ == "__main__":
    main()
