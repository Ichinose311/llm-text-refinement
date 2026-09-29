import argparse
import glob
import json
import os
from ranking_metrics import ranking_metrics



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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--evaluator-dir", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--save-scored", default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        os.makedirs(os.path.dirname(args.save_scored) or ".", exist_ok=True)
        with open(args.save_scored, "w", encoding="utf-8") as f:
            json.dump(scored, f, ensure_ascii=False, indent=2)
        print("\nsaved scored rows to:", args.save_scored)


if __name__ == "__main__":
    main()
