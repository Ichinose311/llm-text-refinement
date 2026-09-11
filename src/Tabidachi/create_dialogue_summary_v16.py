import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_ID = {102, 114, 119, 203, 209, 305, 312, 316}


def extract_user_lines(dialogue: str) -> str:
    lines = []
    for line in dialogue.splitlines():
        line = line.strip()
        if line.startswith("B:"):
            lines.append(line[2:].strip())
    return " ".join(lines).strip()


def conservative_fallback(dialogue: str) -> Dict[str, Any]:
    user_text = extract_user_lines(dialogue)
    if len(user_text) > 500:
        user_text = user_text[:500] + "..."
    return {
        "destination": [],
        "season": [],
        "companions": [],
        "interests": [],
        "constraints": [],
        "food": [],
        "raw_summary": "ユーザ発話: " + user_text,
        "fallback": True,
    }


def build_prompt(dialogue: str) -> str:
    return f"""あなたは観光旅行相談対話から、ユーザBの旅行希望だけを抽出する要約器です。

必ず守ること:
- 対話に明示されている情報だけを書く。
- 対話にない地名、施設名、趣味、同行者、季節、食べ物を作らない。
- 推薦候補の情報は書かない。
- オペレータAの提案ではなく、ユーザBの希望を中心に書く。
- 出力はJSONのみ。説明文やMarkdownは出力しない。

出力形式:
{{
  "destination": ["地名。なければ空配列"],
  "season": ["季節。なければ空配列"],
  "companions": ["同行者。なければ空配列"],
  "interests": ["目的・興味・見たいもの。なければ空配列"],
  "constraints": ["条件・制約。なければ空配列"],
  "food": ["食べたいもの。なければ空配列"],
  "raw_summary": "ユーザBの希望だけを1文で要約"
}}

対話:
{dialogue}

JSON:
"""


def build_raw_summary_from_json(obj: Dict[str, Any]) -> str:
    parts = []

    if obj.get("destination"):
        parts.append("行き先は" + "、".join(map(str, obj["destination"])))
    if obj.get("season"):
        parts.append("時期は" + "、".join(map(str, obj["season"])))
    if obj.get("companions"):
        parts.append("同行者は" + "、".join(map(str, obj["companions"])))
    if obj.get("interests"):
        parts.append("目的・興味は" + "、".join(map(str, obj["interests"])))
    if obj.get("constraints"):
        parts.append("条件は" + "、".join(map(str, obj["constraints"])))
    if obj.get("food"):
        parts.append("食べたいものは" + "、".join(map(str, obj["food"])))

    if not parts:
        return ""

    return "ユーザは、" + "。".join(parts) + "。"


def extract_json(text: str) -> Dict[str, Any] | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    list_keys = [
        "destination",
        "season",
        "companions",
        "interests",
        "constraints",
        "food",
    ]

    for key in list_keys:
        if key not in obj:
            obj[key] = []
        elif not isinstance(obj[key], list):
            obj[key] = [str(obj[key])]

    if "raw_summary" not in obj or not str(obj["raw_summary"]).strip():
        obj["raw_summary"] = build_raw_summary_from_json(obj)
    else:
        obj["raw_summary"] = str(obj["raw_summary"]).strip()

    obj["fallback"] = False
    return obj


def generate_summary(model, tokenizer, dialogue: str, max_new_tokens: int) -> Dict[str, Any]:
    prompt = build_prompt(dialogue)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    parsed = extract_json(generated)
    if parsed is None or not parsed.get("raw_summary"):
        parsed = conservative_fallback(dialogue)
        parsed["raw_model_output"] = generated
    else:
        parsed["raw_model_output"] = generated

    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create strict dialogue summaries for Tabidachi using a configurable LLM."
    )
    parser.add_argument(
        "--input-dir",
        default=os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_1"),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory, e.g. data/Tabidachi/dialogue_summary_v16_test",
    )
    parser.add_argument(
        "--model-name",
        default="tokyotech-llm/Llama-3.1-Swallow-8B-v0.1",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--test-ids-only", action="store_true")
    parser.add_argument("--save-dialogue", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    selected_files = []

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

        selected_files.append(path)

    if args.max_files is not None:
        selected_files = selected_files[: args.max_files]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    processed = 0

    for path in tqdm(selected_files, desc="files"):
        if args.limit is not None and processed >= args.limit:
            break

        filename = os.path.basename(path)

        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        output_items: List[Dict[str, Any]] = []
        seen_dialogues = set()

        for data_id, item in enumerate(tqdm(dataset, desc=filename, leave=False)):
            if args.limit is not None and processed >= args.limit:
                break

            dialogue = item["dialogue"]
            dialogue_key = dialogue[:500]
            if dialogue_key in seen_dialogues:
                continue
            seen_dialogues.add(dialogue_key)

            summary = generate_summary(
                model=model,
                tokenizer=tokenizer,
                dialogue=dialogue,
                max_new_tokens=args.max_new_tokens,
            )

            record = {
                "source_file": filename,
                "data_id": data_id,
                "dialogue_summary": summary["raw_summary"],
                "summary_json": summary,
            }

            if args.save_dialogue:
                record["dialogue"] = dialogue

            output_items.append(record)
            processed += 1

        output_path = os.path.join(args.output, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_items, f, ensure_ascii=False, indent=4)

        print(f"wrote {output_path}: {len(output_items)} summaries")

    print(f"done: processed {processed} dialogues")


if __name__ == "__main__":
    main()
