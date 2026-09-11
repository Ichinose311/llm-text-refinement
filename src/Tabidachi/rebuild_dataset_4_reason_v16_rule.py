import argparse
import glob
import json
import os
from typing import Any, Dict, List


def get_reason_by_source(row: Dict[str, Any], source: str) -> str | None:
    for sample in row.get("all_samples", []):
        if sample.get("source") == source:
            return sample.get("reason")
    return None


def rebuild_row(row: Dict[str, Any]) -> Dict[str, Any]:
    score = int(row["score"])

    v16_reason = get_reason_by_source(row, "v16_sample_0")
    generic_neutral = get_reason_by_source(row, "generic_neutral")
    generic_negative = get_reason_by_source(row, "generic_negative")

    if v16_reason is None:
        v16_reason = row["chosen_recommendation_sentence"]

    if generic_neutral is None:
        generic_neutral = (
            "推薦する理由: 明確な推薦理由はありません。\n"
            "推薦しない理由: 明確な推薦しない理由はありません。"
        )

    if generic_negative is None:
        generic_negative = (
            "推薦する理由: 明確な推薦理由はありません。\n"
            "推薦しない理由: この候補はユーザ要約との対応が限定的です。"
        )

    if score == 1:
        chosen = v16_reason
        rejected = generic_neutral
        chosen_source = "v16_sample_0"
        rejected_source = "generic_neutral"
    else:
        chosen = generic_negative
        rejected = v16_reason
        chosen_source = "generic_negative"
        rejected_source = "v16_sample_0"

    new_row = dict(row)
    new_row["chosen_recommendation_sentence"] = chosen
    new_row["rejected_recommendation_sentence"] = rejected
    new_row["chosen_source_rule"] = chosen_source
    new_row["rejected_source_rule"] = rejected_source

    return new_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    total = 0
    same = 0

    for path in sorted(glob.glob(os.path.join(args.input_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)

        rebuilt: List[Dict[str, Any]] = []
        for row in rows:
            new_row = rebuild_row(row)
            rebuilt.append(new_row)
            total += 1
            if new_row["chosen_recommendation_sentence"] == new_row["rejected_recommendation_sentence"]:
                same += 1

        out_path = os.path.join(args.output, os.path.basename(path))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rebuilt, f, ensure_ascii=False, indent=4)

        print(f"wrote {out_path}: {len(rebuilt)} rows")

    print("total:", total)
    print("same_pairs:", same)


if __name__ == "__main__":
    main()
