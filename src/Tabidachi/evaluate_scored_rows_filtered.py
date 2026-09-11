import argparse
import json
import math
from collections import defaultdict


def dcg(labels, k):
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(labels[:k]))


def evaluate(rows, ks):
    groups = defaultdict(list)
    for x in rows:
        groups[(x["source_file"], int(x["data_id"]))].append(x)

    metrics = {}
    for k in ks:
        recall = []
        ndcg = []
        mrr = []

        for items in groups.values():
            ranked = sorted(items, key=lambda x: float(x["predicted_score"]), reverse=True)
            labels = [int(x["score"]) for x in ranked]
            topk = labels[:k]

            recall.append(1.0 if any(v > 0 for v in topk) else 0.0)

            rr = 0.0
            for i, v in enumerate(topk, start=1):
                if v > 0:
                    rr = 1.0 / i
                    break
            mrr.append(rr)

            ideal = sorted(labels, reverse=True)
            idcg = dcg(ideal, k)
            ndcg.append(dcg(labels, k) / idcg if idcg > 0 else 0.0)

        metrics[f"Recall@{k}"] = sum(recall) / len(recall)
        metrics[f"NDCG@{k}"] = sum(ndcg) / len(ndcg)
        metrics[f"MRR@{k}"] = sum(mrr) / len(mrr)

    return metrics, len(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-json", required=True)
    parser.add_argument("--filter-dir", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--save-json")
    args = parser.parse_args()

    import glob, os
    filter_files = {os.path.basename(p) for p in glob.glob(os.path.join(args.filter_dir, "*.json"))}

    with open(args.scored_json, encoding="utf-8") as f:
        rows = json.load(f)

    rows = [x for x in rows if x["source_file"] in filter_files]

    metrics, group_count = evaluate(rows, args.ks)

    print("scored:", args.scored_json)
    print("filter:", args.filter_dir)
    print("rows:", len(rows))
    print("groups:", group_count)

    print("\n=== Ranking Metrics ===")
    for k in sorted(metrics):
        print(f"{k}: {metrics[k]:.6f}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({
                "scored_json": args.scored_json,
                "filter_dir": args.filter_dir,
                "rows": len(rows),
                "groups": group_count,
                "metrics": metrics,
            }, f, ensure_ascii=False, indent=2)
        print("\nsaved:", args.save_json)


if __name__ == "__main__":
    main()
