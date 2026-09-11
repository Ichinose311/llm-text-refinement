import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple

import torch
from safetensors.torch import load_file
from tqdm import tqdm
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


def build_candidate_information(candidate: Dict[str, Any]) -> str:
    summary = candidate.get("Summary", "") or ""
    feature = candidate.get("Feature", "") or ""
    return (summary + " " + feature).strip()


def load_summary_map(summary_dir: str) -> Dict[Tuple[str, int], str]:
    summary_map: Dict[Tuple[str, int], str] = {}

    for path in glob.glob(os.path.join(summary_dir, "*.json")):
        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            key = (record["source_file"], int(record["data_id"]))
            summary_map[key] = record["dialogue_summary"]

    return summary_map


def load_baseline2_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DebertaRegressionModel(MODEL_NAME)

    model_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")

    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


def predict_score(tokenizer, model, device, dialogue_summary: str, candidate_information: str, max_length: int) -> float:
    sep = tokenizer.sep_token
    text = f"{dialogue_summary}{sep}{candidate_information}"

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--model-dir", default="deberta_best_model_baseline2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    summary_map = load_summary_map(args.summary_dir)
    tokenizer, model, device = load_baseline2_model(args.model_dir)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    files = [
        f for f in files
        if len(os.path.basename(f)) > 4 and os.path.basename(f)[4] in {"1", "2", "3"}
    ]

    if args.max_files is not None:
        files = files[: args.max_files]

    total_rows = 0
    processed_groups = 0

    for file_path in tqdm(files, desc="files"):
        filename = os.path.basename(file_path)
        output_path = os.path.join(args.output, filename)

        with open(file_path, encoding="utf-8") as f:
            dataset = json.load(f)

        output_rows: List[Dict[str, Any]] = []

        for data_id, data in enumerate(dataset):
            key = (filename, data_id)
            if key not in summary_map:
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
                pred = predict_score(
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    dialogue_summary=dialogue_summary,
                    candidate_information=candidate_information,
                    max_length=args.max_length,
                )

                output_rows.append({
                    "dialogue_summary": dialogue_summary,
                    "candidate_information": candidate_information,
                    "recommendation_sentence": "",
                    "predicted_score": pred,
                    "score": int(scores[candidate_index]),
                    "source_file": filename,
                    "data_id": data_id,
                    "candidate_index": candidate_index,
                })

            processed_groups += 1

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_rows, f, ensure_ascii=False, indent=4)

        total_rows += len(output_rows)
        print(f"wrote {output_path}: {len(output_rows)} rows")

    print("processed_groups:", processed_groups)
    print("total_rows:", total_rows)


if __name__ == "__main__":
    main()
