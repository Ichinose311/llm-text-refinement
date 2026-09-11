import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from oss_llm_v16 import Llama_Swallow


def build_candidate_information(candidate: Dict[str, Any]) -> str:
    summary = candidate.get("Summary", "") or ""
    feature = candidate.get("Feature", "") or ""
    return (summary + " " + feature).strip()


def load_summary_map(summary_dir: str) -> Dict[Tuple[str, int], str]:
    summary_map: Dict[Tuple[str, int], str] = {}

    for path in glob.glob(os.path.join(summary_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            key = (record["source_file"], int(record["data_id"]))
            summary_map[key] = record["dialogue_summary"]

    return summary_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate recommendation reasons with v16 prompt and DPO LoRA adapter."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    summary_map = load_summary_map(args.summary_dir)
    if not summary_map:
        raise ValueError(f"No summaries found in: {args.summary_dir}")

    llm = Llama_Swallow(adapter_dir=args.adapter_dir)

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

                recommendation_sentence = llm.generate_rec_reason_with_dialogue(
                    dialogue_summary,
                    candidate_information,
                    temperature=args.temperature,
                )

                output_rows.append({
                    "dialogue_summary": dialogue_summary,
                    "candidate_information": candidate_information,
                    "recommendation_sentence": recommendation_sentence,
                    "score": int(scores[candidate_index]),
                    "source_file": filename,
                    "data_id": data_id,
                    "candidate_index": candidate_index,
                })

            processed_dialogues += 1

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_rows, f, ensure_ascii=False, indent=4)

        print(f"wrote {output_path}: {len(output_rows)} rows")

    print(f"done: processed {processed_dialogues} dialogue examples")


if __name__ == "__main__":
    main()
