import argparse
import glob
import inspect
import json
import os
import sys

from tqdm import tqdm

from oss_llm_v16 import Llama_Swallow
import create_dialogue_summary_v16 as base


CANDIDATE_FUNCS = [
    "generate_dialogue_summary",
    "generate_summary_with_v16_prompt",
    "generate_one_summary",
    "create_dialogue_summary",
    "summarize_dialogue",
    "generate_summary",
]


def find_summary_func():
    for name in CANDIDATE_FUNCS:
        fn = getattr(base, name, None)
        if callable(fn):
            print("using summary function:", name)
            return fn

    funcs = [
        name for name, obj in vars(base).items()
        if callable(obj) and not name.startswith("_")
    ]
    print("summary function not found.")
    print("available functions:", funcs)
    sys.exit(1)


def call_summary(fn, llm, dialogue, temperature, max_new_tokens):
    sig = inspect.signature(fn)
    params = sig.parameters

    kwargs = {}
    args = []

    if "llm" in params:
        kwargs["llm"] = llm
    elif "model" in params:
        kwargs["model"] = getattr(llm, "model", llm)

    if "tokenizer" in params:
        kwargs["tokenizer"] = getattr(llm, "tokenizer", None)

    if "max_new_tokens" in params:
        kwargs["max_new_tokens"] = max_new_tokens

    if "dialogue" in params:
        kwargs["dialogue"] = dialogue
    elif "text" in params:
        kwargs["text"] = dialogue
    else:
        args.append(dialogue)

    if "temperature" in params:
        kwargs["temperature"] = temperature

    result = fn(*args, **kwargs)

    if isinstance(result, dict):
        dialogue_summary = (
            result.get("dialogue_summary")
            or result.get("raw_summary")
            or result.get("summary")
            or ""
        )
        summary_json = result.get("summary_json", result)
        return dialogue_summary, summary_json

    if isinstance(result, tuple) and len(result) >= 1:
        dialogue_summary = str(result[0])
        return dialogue_summary, {"raw_summary": dialogue_summary, "raw_model_output": result}

    dialogue_summary = str(result)
    return dialogue_summary, {
        "raw_summary": dialogue_summary,
        "fallback": False,
        "raw_model_output": dialogue_summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-dialogue", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if args.max_files is not None:
        files = files[:args.max_files]

    fn = find_summary_func()
    llm = Llama_Swallow()

    total = 0

    for path in tqdm(files, desc="files"):
        source_file = os.path.basename(path)
        out_path = os.path.join(args.output, source_file)

        if os.path.exists(out_path) and not args.overwrite:
            print("skip existing:", out_path)
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if args.limit is not None:
            data = data[:args.limit]

        rows = []
        for data_id, row in enumerate(tqdm(data, desc=source_file, leave=False)):
            dialogue = row.get("dialogue", "")
            if not dialogue:
                continue

            dialogue_summary, summary_json = call_summary(
                fn, llm, dialogue, args.temperature, args.max_new_tokens
            )

            out = {
                "source_file": source_file,
                "data_id": data_id,
                "dialogue_summary": dialogue_summary,
                "summary_json": summary_json,
            }

            if args.save_dialogue:
                out["dialogue"] = dialogue

            rows.append(out)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        total += len(rows)
        print(f"wrote {out_path}: {len(rows)} summaries")

    print("done: processed", total, "groups")


if __name__ == "__main__":
    main()
