import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List

from tqdm import tqdm

from oss_llm_v16 import Llama_Swallow


FIELDS = ["destination", "season", "companions", "interests", "constraints", "food"]


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def extract_json(text: str) -> Dict[str, Any] | None:
    if not text:
        return None

    text = text.strip()
    candidates = [text]

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def fallback_summary(dialogue: str) -> str:
    b_utts = []
    for line in str(dialogue).splitlines():
        line = line.strip()
        if line.startswith("B:"):
            b_utts.append(line[2:].strip())

    text = " ".join(b_utts)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "ユーザの旅行希望は明確ではない。"
    return "ユーザ発話: " + text[:220]


def build_raw_summary(obj: Dict[str, Any], dialogue: str) -> str:
    raw = str(obj.get("raw_summary", "")).strip()
    if raw:
        return raw

    parts = []

    destination = as_list(obj.get("destination"))
    season = as_list(obj.get("season"))
    companions = as_list(obj.get("companions"))
    interests = as_list(obj.get("interests"))
    constraints = as_list(obj.get("constraints"))
    food = as_list(obj.get("food"))

    if destination:
        parts.append("・".join(destination) + "に行きたい")
    if season:
        parts.append("時期は" + "・".join(season))
    if companions:
        parts.append("同行者は" + "・".join(companions))
    if interests:
        parts.append("関心は" + "・".join(interests))
    if constraints:
        parts.append("条件は" + "・".join(constraints))
    if food:
        parts.append("食の希望は" + "・".join(food))

    if parts:
        return "、".join(parts) + "。"

    return fallback_summary(dialogue)


def parse_summary(raw_model_output: str, dialogue: str) -> Dict[str, Any]:
    obj = extract_json(raw_model_output)

    if obj is None:
        raw_summary = fallback_summary(dialogue)
        return {
            "dialogue_summary": raw_summary,
            "summary_json": {
                "destination": [],
                "season": [],
                "companions": [],
                "interests": [],
                "constraints": [],
                "food": [],
                "raw_summary": raw_summary,
                "fallback": True,
                "raw_model_output": raw_model_output,
            },
        }

    summary_json = {}
    for field in FIELDS:
        summary_json[field] = as_list(obj.get(field))

    raw_summary = build_raw_summary(obj, dialogue)
    summary_json["raw_summary"] = raw_summary
    summary_json["fallback"] = False
    summary_json["raw_model_output"] = raw_model_output

    return {
        "dialogue_summary": raw_summary,
        "summary_json": summary_json,
    }


def build_prompt(dialogue: str) -> str:
    return f"""以下の観光相談対話から、B（ユーザ）の旅行希望だけを抽出してください。
A（オペレータ）の提案内容や、対話に出ていない情報は入れないでください。

出力は必ずJSONのみとし、説明文やMarkdownは出力しないでください。

抽出項目:
- destination: 行きたい地域、都道府県、方面
- season: 希望時期、季節
- companions: 同行者、人数
- interests: 見たいもの、体験したいこと、関心
- constraints: 移動、予算、時間、年齢、設備などの条件
- food: 食べ物や飲食に関する希望
- raw_summary: 上記を自然な日本語1文で短くまとめた文

値がない項目は空配列 [] にしてください。

出力形式:
{{
  "destination": [],
  "season": [],
  "companions": [],
  "interests": [],
  "constraints": [],
  "food": [],
  "raw_summary": ""
}}

対話:
{dialogue}
"""


def generate_summary_with_v16_prompt(llm: Llama_Swallow, dialogue: str, temperature: float) -> Dict[str, Any]:
    prompt = build_prompt(dialogue)

    # Llama_Swallow has no public method for this custom v16 JSON prompt.
    raw_output = llm._Llama_Swallow__generate_text(prompt, temperature=temperature)

    return parse_summary(raw_output, dialogue)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-dialogue", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if args.max_files is not None:
        files = files[: args.max_files]

    llm = Llama_Swallow()

    total = 0

    for path in tqdm(files, desc="files"):
        source_file = os.path.basename(path)
        out_path = os.path.join(args.output, source_file)

        if os.path.exists(out_path) and not args.overwrite:
            print(f"skip existing: {out_path}")
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if args.limit is not None:
            data = data[: args.limit]

        output_rows = []

        for data_id, row in enumerate(tqdm(data, desc=source_file, leave=False)):
            dialogue = row.get("dialogue", "")
            if not dialogue:
                continue

            summary = generate_summary_with_v16_prompt(
                llm,
                dialogue,
                temperature=args.temperature,
            )

            output = {
                "source_file": source_file,
                "data_id": data_id,
                "dialogue_summary": summary["dialogue_summary"],
                "summary_json": summary["summary_json"],
            }

            if args.save_dialogue:
                output["dialogue"] = dialogue

            output_rows.append(output)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_rows, f, ensure_ascii=False, indent=2)

        total += len(output_rows)
        print(f"wrote {out_path}: {len(output_rows)} summaries")

    print(f"done: processed {total} groups")


if __name__ == "__main__":
    main()
