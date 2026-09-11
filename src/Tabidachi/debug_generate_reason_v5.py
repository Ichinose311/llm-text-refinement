import os
import json
from oss_llm import Llama_Swallow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_file = os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_1/102_1_1_full.json")

MAX_CASES = 4          # まず4事例だけ見る
NEG_PER_CASE = 2       # 各事例で負例2件を見る

with open(input_file, "r", encoding="utf-8") as f:
    datas = json.load(f)

llm = Llama_Swallow()

for case_id, data in enumerate(datas[:MAX_CASES], start=1):
    print("\n" + "#" * 100)
    print(f"事例 {case_id}")
    print("#" * 100)

    print("===== 対話要約を生成中 =====")
    dialogue_summary = llm.generate_summary(data["dialogue"])

    # 表示用にも不要語を消す
    clean_summary = (
        dialogue_summary
        .replace("##", "")
        .replace("解答例", "")
        .replace("Bは、", "ユーザは、")
        .replace("Bは", "ユーザは")
        .strip()
    )

    print("\n===== 対話要約 =====")
    print(clean_summary)

    positive_indices = [i for i, s in enumerate(data["score"]) if s == 1]
    negative_indices = [i for i, s in enumerate(data["score"]) if s == 0]

    target_indices = positive_indices[:1] + negative_indices[:NEG_PER_CASE]

    print(f"\n表示する候補index: {target_indices}")

    for i in target_indices:
        candidate = data["candidates"][i]
        candidate_info = candidate["Summary"] + candidate["Feature"]
        gold_score = data["score"][i]

        print("\n" + "=" * 80)
        print(f"候補 {i+1}")
        print(f"正解ラベル: {gold_score}")

        name = (
            candidate.get("Name")
            or candidate.get("name")
            or candidate.get("SpotName")
            or candidate.get("spot_name")
            or candidate.get("Title")
            or candidate.get("title")
            or ""
        )

        if name:
            print(f"観光地名: {name}")

        print("\n--- 観光地情報 ---")
        print(candidate_info[:500])

        print("\n--- 生成された推薦理由・非推薦理由 ---")
        reason = llm.generate_rec_reason_with_dialogue(
            clean_summary,
            candidate_info,
            temperature=0.1
        )
        print(reason)
