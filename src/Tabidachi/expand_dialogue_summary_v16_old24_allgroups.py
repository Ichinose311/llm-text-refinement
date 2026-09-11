import argparse
import glob
import json
import os
from tqdm import tqdm

TEST_IDS = {102, 114, 119, 203, 209, 305, 312, 316}


def is_target_file(path):
    name = os.path.basename(path)
    try:
        return int(name[:3]) in TEST_IDS and name[4] in {"1", "2", "3"}
    except Exception:
        return False


def norm_text(text):
    return " ".join(str(text or "").split())


def load_summary_entries(summary_dir, filename):
    path = os.path.join(summary_dir, filename)
    if not os.path.exists(path):
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return data if isinstance(data, list) else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/Tabidachi/datasets_1")
    parser.add_argument("--summary-dir", default="data/Tabidachi/dialogue_summary_v16_all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-dialogue", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = [p for p in sorted(glob.glob(os.path.join(args.input_dir, "*.json"))) if is_target_file(p)]
    if args.max_files is not None:
        files = files[: args.max_files]

    total = 0
    skipped = 0

    for path in tqdm(files, desc="files"):
        filename = os.path.basename(path)
        out_path = os.path.join(args.output, filename)

        if os.path.exists(out_path) and not args.overwrite:
            print(f"skip existing: {out_path}")
            continue

        with open(path, encoding="utf-8") as f:
            rows = json.load(f)

        summaries = load_summary_entries(args.summary_dir, filename)

        by_data_id = {}
        by_dialogue = {}
        first_summary = None

        for s in summaries:
            if "dialogue_summary" not in s:
                continue
            if first_summary is None:
                first_summary = s["dialogue_summary"]
            if "data_id" in s:
                by_data_id[int(s["data_id"])] = s["dialogue_summary"]
            if "dialogue" in s:
                by_dialogue[norm_text(s["dialogue"])] = s["dialogue_summary"]

        output_rows = []
        use_rows = rows[: args.limit] if args.limit is not None else rows

        for data_id, row in enumerate(use_rows):
            dialogue = row.get("dialogue", "")
            summary = by_dialogue.get(norm_text(dialogue))
            source = "dialogue"

            if summary is None:
                summary = by_data_id.get(data_id)
                source = "data_id"

            if summary is None:
                summary = first_summary
                source = "file_fallback"

            if summary is None:
                skipped += 1
                print(f"summary not found, skipped: {(filename, data_id)}")
                continue

            item = {
                "source_file": filename,
                "data_id": data_id,
                "dialogue_summary": summary,
                "summary_source": source,
            }
            if args.save_dialogue:
                item["dialogue"] = dialogue

            output_rows.append(item)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_rows, f, ensure_ascii=False, indent=2)

        total += len(output_rows)
        print(f"wrote {out_path}: {len(output_rows)} summaries")

    print("done")
    print("summaries:", total)
    print("skipped:", skipped)


if __name__ == "__main__":
    main()
