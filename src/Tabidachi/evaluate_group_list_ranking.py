import argparse
import glob
import json
import math
import os


def dcg_at_k(labels, k):
    total = 0.0
    for i, rel in enumerate(labels[:k], start=1):
        total += (2 ** rel - 1) / math.log2(i + 1)
    return total


def evaluate_group(predicted_score, score, ks):
    pairs = list(zip(predicted_score, score))
    pairs.sort(key=lambda x: x[0], reverse=True)

    ranked_labels = [int(x[1]) for x in pairs]
    ideal_labels = sorted([int(x) for x in score], reverse=True)

    result = {}

    for k in ks:
        topk = ranked_labels[:k]

        result[f"Recall@{k}"] = 1.0 if any(x > 0 for x in topk) else 0.0

        rr = 0.0
        for i, rel in enumerate(topk, start=1):
            if rel > 0:
                rr = 1.0 / i
                break
        result[f"MRR@{k}"] = rr

        dcg = dcg_at_k(ranked_labels, k)
        idcg = dcg_at_k(ideal_labels, k)
        result[f"NDCG@{k}"] = dcg / idcg if idcg > 0 else 0.0

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--save-json")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.input_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for i, row in enumerate(data):
            pred = row.get("predicted_score")
            score = row.get("score")
            if not isinstance(pred, list) or not isinstance(score, list):
                continue
            if len(pred) != len(score):
                continue
            if len(score) == 0:
                continue
            rows.append({
                "source_file": os.path.basename(path),
                "row_index": i,
                "predicted_score": pred,
                "score": score,
            })

    if not rows:
        raise RuntimeError("no valid rows found")

    metrics_sum = {}
    for row in rows:
        m = evaluate_group(row["predicted_score"], row["score"], args.ks)
        for k, v in m.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v

    metrics = {k: v / len(rows) for k, v in metrics_sum.items()}

    print("input:", args.input_dir)
    print("groups:", len(rows))
    print("\n=== Ranking Metrics ===")
    for k in sorted(metrics):
        print(f"{k}: {metrics[k]:.6f}")

    if args.save_json:
        output = {
            "input_dir": args.input_dir,
            "groups": len(rows),
            "metrics": metrics,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print("\nsaved:", args.save_json)


if __name__ == "__main__":
    main()
