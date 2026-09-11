import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "globis-university/deberta-v3-japanese-large"


class DebertaRegressionModel(torch.nn.Module):
    def __init__(self, pretrained_model_name: str):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(pretrained_model_name)
        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(self.base_model.config.hidden_size, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.regression_head(cls_output).squeeze(-1)
        return logits


def load_evaluator(evaluator_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(evaluator_dir)
    model = DebertaRegressionModel(MODEL_NAME)

    model_path = os.path.join(evaluator_dir, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")

    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


def predict_score(
    tokenizer,
    model,
    device,
    dialogue_summary: str,
    recommendation_sentence: str,
    candidate_information: str,
    max_length: int,
) -> float:
    sep = tokenizer.sep_token
    text = f"{dialogue_summary}{sep}{recommendation_sentence}{sep}{candidate_information}"

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        pred = model(**inputs)

    return float(pred.detach().cpu().numpy().reshape(-1)[0])


def load_rows(input_dir: str) -> List[Dict[str, Any]]:
    rows = []

    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            rows.append(item)

    return rows


def group_rows(rows: List[Dict[str, Any]]):
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row.get("source_file", ""),
            int(row.get("data_id", 0)),
        )
        grouped[key].append(row)

    return grouped


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

        if positives > 0:
            result[f"Recall@{k}"] = hit_count / positives
        else:
            result[f"Recall@{k}"] = 0.0

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


def average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}

    keys = sorted(metrics_list[0].keys())
    return {
        key: float(np.mean([m[key] for m in metrics_list]))
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--evaluator-dir", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--save-scored", default=None)
    args = parser.parse_args()

    tokenizer, model, device = load_evaluator(args.evaluator_dir)

    rows = load_rows(args.input_dir)
    print(f"loaded rows: {len(rows)}")

    if not rows:
        raise ValueError("No rows found.")

    scored_rows = []
    for row in rows:
        pred = predict_score(
            tokenizer=tokenizer,
            model=model,
            device=device,
            dialogue_summary=row["dialogue_summary"],
            recommendation_sentence=row["recommendation_sentence"],
            candidate_information=row["candidate_information"],
            max_length=args.max_length,
        )
        new_row = dict(row)
        new_row["predicted_score"] = pred
        scored_rows.append(new_row)

    grouped = group_rows(scored_rows)
    print(f"groups: {len(grouped)}")

    metrics_list = []

    for key, group in grouped.items():
        ranked = sorted(group, key=lambda x: x["predicted_score"], reverse=True)
        labels = [int(x["score"]) for x in ranked]
        metrics_list.append(metrics_for_group(labels, args.ks))

    avg = average_metrics(metrics_list)

    print("\n=== Ranking Metrics ===")
    for key, value in avg.items():
        print(f"{key}: {value:.6f}")

    if args.save_scored:
        os.makedirs(os.path.dirname(args.save_scored), exist_ok=True)
        with open(args.save_scored, "w", encoding="utf-8") as f:
            json.dump(scored_rows, f, ensure_ascii=False, indent=4)
        print(f"\nsaved scored rows to: {args.save_scored}")


if __name__ == "__main__":
    main()
