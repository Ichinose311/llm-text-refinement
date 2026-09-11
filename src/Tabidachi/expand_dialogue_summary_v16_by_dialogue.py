import argparse
import glob
import json
import os
from collections import Counter


def normalize_text(text):
    return " ".join(str(text).replace("\r\n", "\n").replace("\r", "\n").split())


def load_summary_map(summary_dir):
    summary_map = {}
    duplicate = 0

    for path in sorted(glob.glob(os.path.join(summary_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)

        for row in rows:
            dialogue = row.get("dialogue")
            if not dialogue:
                continue

            key = normalize_text(dialogue)
            if key in summary_map:
                duplicate += 1
                continue

            summary_map[key] = row

    return summary_map, duplicate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-dialogue", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    summary_map, duplicate = load_summary_map(args.summary_dir)
    print("loaded unique summaries:", len(summary_map))
    print("duplicate summaries:", duplicate)

    total_rows = 0
    total_missing = 0
    source_counter = Counter()

    for path in sorted(glob.glob(os.path.join(args.input_dir, "*.json"))):
        source_file = os.path.basename(path)
        out_path = os.path.join(args.output, source_file)

        if os.path.exists(out_path) and not args.overwrite:
            print("skip existing:", out_path)
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        output_rows = []
        missing = 0

        for data_id, row in enumerate(data):
            dialogue = row.get("dialogue", "")
            key = normalize_text(dialogue)

            if key not in summary_map:
                missing += 1
                total_missing += 1
                continue

            summary_row = summary_map[key]

            out = {
                "source_file": source_file,
                "data_id": data_id,
                "dialogue_summary": summary_row.get("dialogue_summary", ""),
                "summary_json": summary_row.get("summary_json", {}),
                "summary_source": "dialogue",
            }

            if args.save_dialogue:
                out["dialogue"] = dialogue

            output_rows.append(out)
            source_counter["dialogue"] += 1

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_rows, f, ensure_ascii=False, indent=2)

        total_rows += len(output_rows)
        print(f"wrote {out_path}: {len(output_rows)} summaries, missing={missing}")

    print("done")
    print("rows:", total_rows)
    print("missing:", total_missing)
    print("sources:", source_counter)


if __name__ == "__main__":
    main()
