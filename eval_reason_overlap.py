"""Heuristic text diagnostics; counts are not factuality judgments."""

import argparse
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "artifacts/generated_text/reason_examples/reason_examples_v16_multi.txt")
    args = parser.parse_args()
    try:
        text = args.input.read_text(encoding="utf-8")
    except OSError as exc:
        parser.error(str(exc))

    blocks = text.split("--- 生成された推薦理由・非推薦理由 ---")

    overlap_count = 0
    pair_count = 0

    for block in blocks[1:]:
        rec_match = re.search(r"推薦する理由:\s*(.*)", block)
        not_match = re.search(r"推薦しない理由:\s*(.*)", block)

        if not rec_match or not not_match:
            continue

        rec = rec_match.group(1).strip()
        not_rec = not_match.group(1).strip()

        rec_terms = set(re.findall(r"「([^」]+)」", rec))
        not_terms = set(re.findall(r"「([^」]+)」", not_rec))

        overlap = rec_terms & not_terms

        pair_count += 1
        if overlap:
            overlap_count += 1
            print("重複あり:")
            print("推薦:", rec)
            print("非推薦:", not_rec)
            print("重複語句:", overlap)
            print()

    print("===== 根拠語句重複評価 =====")
    print(f"評価ペア数: {pair_count}")
    print(f"重複ありペア数: {overlap_count}")


if __name__ == "__main__":
    main()
