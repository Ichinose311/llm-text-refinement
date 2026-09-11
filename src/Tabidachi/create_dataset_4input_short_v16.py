import argparse
import glob
import json
import os
from pathlib import Path


def truncate_chars(text, max_chars):
    text = text or ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def build_short_text(item_sentence, v16_reason, item_max_chars):
    item_sentence = truncate_chars(item_sentence, item_max_chars)
    v16_reason = (v16_reason or "").strip()

    return (
        "既存推薦文:\n"
        f"{item_sentence}\n\n"
        "v16推薦理由:\n"
        f"{v16_reason}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--item-max-chars", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(input_dir.glob("*.json")):
        out_path = output_dir / path.name
        if out_path.exists() and not args.overwrite:
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        output = []
        for x in data:
            y = dict(x)
            y["recommendation_sentence"] = build_short_text(
                x.get("item_recommendation_sentence", ""),
                x.get("v16_reason_sentence", ""),
                args.item_max_chars,
            )
            output.append(y)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"wrote {out_path}: {len(output)} rows")

    print("done")


if __name__ == "__main__":
    main()
