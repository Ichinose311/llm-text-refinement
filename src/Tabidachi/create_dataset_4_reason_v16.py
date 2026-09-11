import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from oss_llm_v16 import Llama_Swallow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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


def build_candidate_information(candidate: Dict[str, Any]) -> str:
    summary = candidate.get("Summary", "") or ""
    feature = candidate.get("Feature", "") or ""
    return (summary + " " + feature).strip()

def build_reason_variants(v16_reasons):
    variants = []
    seen = set()

    def add(source, reason):
        reason = (reason or "").strip()
        if reason and reason not in seen:
            seen.add(reason)
            variants.append({"source": source, "reason": reason})

    for i, reason in enumerate(v16_reasons):
        add(f"v16_sample_{i}", reason)

    add(
        "generic_neutral",
        "推薦する理由: 明確な推薦理由はありません。\n"
        "推薦しない理由: 明確な推薦しない理由はありません。"
    )

    add(
        "generic_positive",
        "推薦する理由: この候補はユーザ要約に合う可能性があります。\n"
        "推薦しない理由: 明確な推薦しない理由はありません。"
    )

    add(
        "generic_negative",
        "推薦する理由: 明確な推薦理由はありません。\n"
        "推薦しない理由: この候補はユーザ要約との対応が限定的です。"
    )

    return variants


def select_preference_pair(scored_samples, gold_score):
    target = float(gold_score)

    ranked = sorted(
        scored_samples,
        key=lambda x: abs(float(x["predicted_score"]) - target)
    )

    chosen = ranked[0]

    rejected = None
    for x in reversed(ranked):
        if x["reason"] != chosen["reason"]:
            rejected = x
            break

    if rejected is None:
        rejected = ranked[-1]

    return chosen, rejected

def load_summary_map(summary_dir: str) -> Dict[Tuple[str, int], str]:
    summary_map: Dict[Tuple[str, int], str] = {}

    for path in glob.glob(os.path.join(summary_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            key = (record["source_file"], int(record["data_id"]))
            summary_map[key] = record["dialogue_summary"]

    return summary_map


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
    max_length: int = 1024,
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


def create_preference_for_candidate(
    llm: Llama_Swallow,
    tokenizer,
    evaluator,
    device,
    dialogue_summary: str,
    candidate_information: str,
    gold_score: int,
    num_samples: int,
    temperature: float,
    max_length: int,
) -> Dict[str, Any]:
    v16_reasons = []
    for sample_id in range(num_samples):
        reason = llm.generate_rec_reason_with_dialogue(
            dialogue_summary,
            candidate_information,
            temperature=temperature,
        )
        v16_reasons.append(reason)

    variants = build_reason_variants(v16_reasons)

    reasons = []
    for sample_id, item in enumerate(variants):
        predicted_score = predict_score(
            tokenizer,
            evaluator,
            device,
            dialogue_summary,
            item["reason"],
            candidate_information,
            max_length=max_length,
        )

        reasons.append({
            "sample_id": sample_id,
            "source": item["source"],
            "reason": item["reason"],
            "predicted_score": float(predicted_score),
        })

    chosen, rejected = select_preference_pair(reasons, gold_score)

    return {
        "prompt": (
            "以下のユーザ要約と観光地情報をもとに、"
            "推薦する理由と推薦しない理由を生成してください。\n\n"
            f"ユーザ要約:\n{dialogue_summary}\n\n"
            f"観光地情報:\n{candidate_information}\n"
        ),
        "dialogue_summary": dialogue_summary,
        "candidate_information": candidate_information,
        "score": int(gold_score),
        "chosen_recommendation_sentence": chosen["reason"],
        "rejected_recommendation_sentence": rejected["reason"],
        "chosen_predicted_score": chosen["predicted_score"],
        "rejected_predicted_score": rejected["predicted_score"],
        "all_samples": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create v16 reason DPO preference data using clean evaluator."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--evaluator-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    summary_map = load_summary_map(args.summary_dir)
    if not summary_map:
        raise ValueError(f"No summaries found in: {args.summary_dir}")

    tokenizer, evaluator, device = load_evaluator(args.evaluator_dir)
    llm = Llama_Swallow()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    files = [
        f for f in files
        if len(os.path.basename(f)) > 4 and os.path.basename(f)[4] in {"1", "2", "3"}
    ]

    if args.max_files is not None:
        files = files[: args.max_files]

    processed_dialogues = 0

    for file_path in tqdm(files, desc="files"):
        if args.limit is not None and processed_dialogues >= args.limit:
            break

        filename = os.path.basename(file_path)
        output_path = os.path.join(args.output, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        output_rows: List[Dict[str, Any]] = []

        for data_id, data in enumerate(tqdm(dataset, desc=filename, leave=False)):
            if args.limit is not None and processed_dialogues >= args.limit:
                break

            key = (filename, data_id)
            if key not in summary_map:
                print(f"summary not found, skipped: {key}")
                continue

            dialogue_summary = summary_map[key]
            candidates = data["candidates"]
            scores = data["score"]
            candidate_limit = (
                len(candidates)
                if args.max_candidates is None
                else min(args.max_candidates, len(candidates))
            )

            for candidate_index in range(candidate_limit):
                candidate_information = build_candidate_information(candidates[candidate_index])
                gold_score = scores[candidate_index]

                pref = create_preference_for_candidate(
                    llm=llm,
                    tokenizer=tokenizer,
                    evaluator=evaluator,
                    device=device,
                    dialogue_summary=dialogue_summary,
                    candidate_information=candidate_information,
                    gold_score=gold_score,
                    num_samples=args.num_samples,
                    temperature=args.temperature,
                    max_length=args.max_length,
                )
                pref["source_file"] = filename
                pref["data_id"] = data_id
                pref["candidate_index"] = candidate_index
                output_rows.append(pref)

            processed_dialogues += 1

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_rows, f, ensure_ascii=False, indent=4)

        print(f"wrote {output_path}: {len(output_rows)} rows")

    print(f"done: processed {processed_dialogues} dialogue examples")


if __name__ == "__main__":
    main()
