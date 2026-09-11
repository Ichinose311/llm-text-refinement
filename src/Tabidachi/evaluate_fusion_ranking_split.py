import argparse
import json
import math
import random
from collections import defaultdict


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def row_key(x):
    return (x["source_file"], int(x["data_id"]), int(x["candidate_index"]))


def group_key(x):
    return (x["source_file"], int(x["data_id"]))


def minmax(values):
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def build_groups(baseline_rows, reason_rows):
    reason_map = {row_key(x): x for x in reason_rows}
    aligned = []

    for b in baseline_rows:
        k = row_key(b)
        if k not in reason_map:
            continue
        r = reason_map[k]
        aligned.append({
            "source_file": b["source_file"],
            "data_id": int(b["data_id"]),
            "candidate_index": int(b["candidate_index"]),
            "score": int(b["score"]),
            "baseline_score": float(b["predicted_score"]),
            "reason_score": float(r["predicted_score"]),
        })

    groups = defaultdict(list)
    for x in aligned:
        groups[group_key(x)].append(x)

    return groups


def rank_group(rows, alpha):
    baseline_norm = minmax([x["baseline_score"] for x in rows])
    reason_norm = minmax([x["reason_score"] for x in rows])

    ranked = []
    for x, bs, rs in zip(rows, baseline_norm, reason_norm):
        y = dict(x)
        y["fusion_score"] = alpha * bs + (1.0 - alpha) * rs
        ranked.append(y)

    ranked.sort(key=lambda x: (-x["fusion_score"], x["candidate_index"]))
    return ranked


def evaluate(groups, group_keys, alpha, ks):
    metrics = {}

    for k in ks:
        recall = []
        mrr = []
        ndcg = []

        for gk in group_keys:
            ranked = rank_group(groups[gk], alpha)
            labels = [x["score"] for x in ranked]

            topk = labels[:k]
            recall.append(1.0 if any(v > 0 for v in topk) else 0.0)

            rr = 0.0
            for i, v in enumerate(topk, start=1):
                if v > 0:
                    rr = 1.0 / i
                    break
            mrr.append(rr)

            dcg = 0.0
            for i, v in enumerate(topk, start=1):
                dcg += (2 ** v - 1) / math.log2(i + 1)

            ideal = sorted(labels, reverse=True)[:k]
            idcg = 0.0
            for i, v in enumerate(ideal, start=1):
                idcg += (2 ** v - 1) / math.log2(i + 1)

            ndcg.append(dcg / idcg if idcg > 0 else 0.0)

        metrics[f"Recall@{k}"] = sum(recall) / len(recall)
        metrics[f"MRR@{k}"] = sum(mrr) / len(mrr)
        metrics[f"NDCG@{k}"] = sum(ndcg) / len(ndcg)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-scored", required=True)
    parser.add_argument("--reason-scored", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--select-metric", default="NDCG@5")
    parser.add_argument("--save-json")
    args = parser.parse_args()

    baseline_rows = load_json(args.baseline_scored)
    reason_rows = load_json(args.reason_scored)
    groups = build_groups(baseline_rows, reason_rows)

    keys = list(groups.keys())
    random.Random(args.seed).shuffle(keys)

    test_size = max(1, int(round(len(keys) * args.test_ratio)))
    test_keys = keys[:test_size]
    dev_keys = keys[test_size:]

    alphas = []
    n = int(round(1.0 / args.step))
    for i in range(n + 1):
        alphas.append(round(i * args.step, 10))

    best_alpha = None
    best_dev = None
    best_value = -1.0

    for alpha in alphas:
        dev_metrics = evaluate(groups, dev_keys, alpha, args.ks)
        value = dev_metrics[args.select_metric]
        if value > best_value:
            best_value = value
            best_alpha = alpha
            best_dev = dev_metrics

    baseline_test = evaluate(groups, test_keys, 1.0, args.ks)
    reason_test = evaluate(groups, test_keys, 0.0, args.ks)
    fusion_test = evaluate(groups, test_keys, best_alpha, args.ks)

    result = {
        "seed": args.seed,
        "groups_total": len(keys),
        "groups_dev": len(dev_keys),
        "groups_test": len(test_keys),
        "select_metric": args.select_metric,
        "best_alpha": best_alpha,
        "dev_best": best_dev,
        "test_baseline_alpha_1": baseline_test,
        "test_reason_alpha_0": reason_test,
        "test_fusion": fusion_test,
    }

    print("groups:", len(keys))
    print("dev groups:", len(dev_keys))
    print("test groups:", len(test_keys))
    print("selected alpha:", best_alpha)
    print("selected by:", args.select_metric)

    print("\n=== Dev Best ===")
    for k, v in sorted(best_dev.items()):
        print(f"{k}: {v:.6f}")

    print("\n=== Test Baseline alpha=1.0 ===")
    for k, v in sorted(baseline_test.items()):
        print(f"{k}: {v:.6f}")

    print("\n=== Test Reason alpha=0.0 ===")
    for k, v in sorted(reason_test.items()):
        print(f"{k}: {v:.6f}")

    print("\n=== Test Fusion ===")
    for k, v in sorted(fusion_test.items()):
        print(f"{k}: {v:.6f}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("\nsaved:", args.save_json)


if __name__ == "__main__":
    main()
