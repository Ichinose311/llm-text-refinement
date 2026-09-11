import argparse
import glob
import json
import math
import os
from collections import defaultdict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_text(x):
    return (
        "ユーザ要約:\n"
        f"{x.get('dialogue_summary', '')}\n\n"
        "観光地情報:\n"
        f"{x.get('candidate_information', '')}\n\n"
        "既存推薦文:\n"
        f"{x.get('item_recommendation_sentence', '')}\n\n"
        "v16推薦理由:\n"
        f"{x.get('v16_reason_sentence', '')}"
    )


def load_rows(input_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.load(f))
    return rows


def ranking_metrics(rows, ks):
    groups = defaultdict(list)
    for x in rows:
        groups[(x["source_file"], x["data_id"])].append(x)

    out = {}
    for k in ks:
        hits = []
        mrrs = []
        ndcgs = []

        for items in groups.values():
            ranked = sorted(items, key=lambda x: float(x["predicted_score"]), reverse=True)
            topk = ranked[:k]
            labels = [int(x["score"]) for x in topk]

            hits.append(1.0 if any(v > 0 for v in labels) else 0.0)

            rr = 0.0
            for i, v in enumerate(labels, start=1):
                if v > 0:
                    rr = 1.0 / i
                    break
            mrrs.append(rr)

            dcg = 0.0
            for i, v in enumerate(labels, start=1):
                if v > 0:
                    dcg += 1.0 / math.log2(i + 1)
            ndcgs.append(dcg)

        out[f"Recall@{k}"] = sum(hits) / len(hits)
        out[f"MRR@{k}"] = sum(mrrs) / len(mrrs)
        out[f"NDCG@{k}"] = sum(ndcgs) / len(ndcgs)

    return out, len(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--evaluator-dir", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--save-scored", default=None)
    args = parser.parse_args()

    rows = load_rows(args.input_dir)
    print("loaded rows:", len(rows))

    tokenizer = AutoTokenizer.from_pretrained(args.evaluator_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.evaluator_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    scores = []
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        texts = [build_text(x) for x in batch]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits.squeeze(-1)

        scores.extend(logits.detach().cpu().float().tolist())

    scored = []
    for x, score in zip(rows, scores):
        y = dict(x)
        y["predicted_score"] = float(score)
        scored.append(y)

    metrics, n_groups = ranking_metrics(scored, args.ks)
    print("groups:", n_groups)
    print("\n=== Ranking Metrics ===")
    for k in sorted(metrics):
        print(f"{k}: {metrics[k]:.6f}")

    if args.save_scored:
        os.makedirs(os.path.dirname(args.save_scored), exist_ok=True)
        with open(args.save_scored, "w", encoding="utf-8") as f:
            json.dump(scored, f, ensure_ascii=False, indent=2)
        print("\nsaved scored rows to:", args.save_scored)


if __name__ == "__main__":
    main()
