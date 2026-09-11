import argparse
import glob
import json
import os
from tqdm import tqdm
from oss_llm_v16 import Llama_Swallow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason-data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.reason_data_dir, "*.json")))
    if args.max_files is not None:
        files = files[:args.max_files]

    llm = Llama_Swallow()

    total = 0

    for path in tqdm(files, desc="files"):
        filename = os.path.basename(path)
        out_path = os.path.join(args.output, filename)

        if os.path.exists(out_path) and not args.overwrite:
            print(f"skip existing: {out_path}")
            continue

        with open(path, encoding="utf-8") as f:
            rows = json.load(f)

        if args.max_rows is not None:
            rows = rows[:args.max_rows]

        output_rows = []

        for row in tqdm(rows, desc=filename, leave=False):
            dialogue_summary = row["dialogue_summary"]
            candidate_information = row["candidate_information"]
            v16_reason = row["recommendation_sentence"]

            rec_sentence = llm.generate_rec_sentence_with_no_dialogue(
                candidate_information,
                temperature=args.temperature,
            )

            combined_sentence = (
                f"推薦文: {rec_sentence}\n"
                f"推薦理由・非推薦理由: {v16_reason}"
            )

            output_rows.append({
                "dialogue_summary": dialogue_summary,
                "candidate_information": candidate_information,
                "item_recommendation_sentence": rec_sentence,
                "v16_reason_sentence": v16_reason,
                "recommendation_sentence": combined_sentence,
                "score": row["score"],
                "source_file": row["source_file"],
                "data_id": row["data_id"],
                "candidate_index": row["candidate_index"],
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_rows, f, ensure_ascii=False, indent=2)

        total += len(output_rows)
        print(f"wrote {out_path}: {len(output_rows)} rows")

    print("done")
    print("rows:", total)


if __name__ == "__main__":
    main()
