"""Evaluate saved candidate scores without GPUs, weights or dependencies."""

import argparse
import json
from pathlib import Path

from ranking_metrics import dcg, metrics_for_group, ranking_metrics


def load_rows(input_dir):
    directory = Path(input_dir)
    if not directory.is_dir():
        raise ValueError(f"Input directory does not exist: {directory}")
    rows = []
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list of candidate rows: {path}")
        rows.extend(data)
    return rows


def save_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--score-key", default="predicted_score")
    parser.add_argument("--save-scored", default=None, help="Optional copy of input rows")
    parser.add_argument("--save-json", default=None, help="Optional metrics report")
    args = parser.parse_args()
    try:
        rows = load_rows(args.input_dir)
        metrics, count = ranking_metrics(rows, args.ks, args.score_key)
        print("loaded rows:", len(rows))
        print("groups:", count)
        print("\n=== Ranking Metrics ===")
        for key in sorted(metrics):
            print(f"{key}: {metrics[key]:.6f}")
        if args.save_scored:
            save_json(args.save_scored, rows)
        if args.save_json:
            save_json(args.save_json, {
                "schema_version": 1, "groups": count, "rows": len(rows),
                "score_key": args.score_key, "ks": args.ks,
                "tie_policy": "stable input order", "ndcg_gain": "linear",
                "zero_positive_policy": "include as zero", "metrics": metrics,
            })
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
