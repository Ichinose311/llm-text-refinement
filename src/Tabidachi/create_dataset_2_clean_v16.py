import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from oss_llm_v16 import Llama_Swallow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_ID = {102, 114, 119, 203, 209, 305, 312, 316}


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


def create_entries(
    llm: Llama_Swallow,
    data: Dict[str, Any],
    dialogue_summary: str,
    source_file: str,
    data_id: int,
    max_candidates: int | None,
    temperature: float,
) -> List[Dict[str, Any]]:
    candidates = data["candidates"]
    scores = data["score"]

    entries = []
    candidate_limit = (
        len(candidates) if max_candidates is None else min(max_candidates, len(candidates))
    )

    for candidate_index in range(candidate_limit):
        candidate = candidates[candidate_index]
        candidate_information = build_candidate_information(candidate)
        score = scores[candidate_index]

        reason = llm.generate_rec_reason_with_dialogue(
            dialogue_summary,
            candidate_information,
            temperature=temperature,
        )

        entries.append(
            {
                "dialogue_summary": dialogue_summary,
                "candidate_information": candidate_information,
                "recommendation_sentence": reason,
                "score": score,
                "source_file": source_file,
                "data_id": data_id,
                "candidate_index": candidate_index,
            }
        )

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create clean Tabidachi datasets_2 using precomputed strict v16 summaries."
    )
    parser.add_argument(
        "--input-dir",
        default=os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_1"),
    )
    parser.add_argument(
        "--summary-dir",
        required=True,
        help="Directory created by create_dialogue_summary_v16.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory, e.g. data/Tabidachi/datasets_2_clean_v16_test",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of dialogue examples to process in total.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Maximum candidates per dialogue. Use small value for smoke tests.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--test-ids-only",
        action="store_true",
        help="Use only Tabidachi test IDs.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    summary_map = load_summary_map(args.summary_dir)
    if not summary_map:
        raise ValueError(f"No summaries found in: {args.summary_dir}")

    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    filtered_files = []
    for path in files:
        name = os.path.basename(path)
        if not (len(name) > 4 and name[4] in {"1", "2", "3"}):
            continue

        if args.test_ids_only:
            try:
                uid = int(name[:3])
            except ValueError:
                continue
            if uid not in TEST_ID:
                continue

        filtered_files.append(path)

    if args.max_files is not None:
        filtered_files = filtered_files[: args.max_files]

    llm = Llama_Swallow()
    processed_dialogues = 0

    for file_path in tqdm(filtered_files, desc="files"):
        if args.limit is not None and processed_dialogues >= args.limit:
            break

        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        output_entries = []

        for data_id, data in enumerate(tqdm(dataset, desc=filename, leave=False)):
            if args.limit is not None and processed_dialogues >= args.limit:
                break

            key = (filename, data_id)
            if key not in summary_map:
                print(f"summary not found, skipped: {key}")
                continue

            dialogue_summary = summary_map[key]

            output_entries.extend(
                create_entries(
                    llm=llm,
                    data=data,
                    dialogue_summary=dialogue_summary,
                    source_file=filename,
                    data_id=data_id,
                    max_candidates=args.max_candidates,
                    temperature=args.temperature,
                )
            )
            processed_dialogues += 1

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_entries, f, ensure_ascii=False, indent=4)

        print(f"wrote {output_path}: {len(output_entries)} entries")

    print(f"done: processed {processed_dialogues} dialogue examples")


if __name__ == "__main__":
    main()