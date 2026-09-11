import os
import json
from oss_llm import Llama_Swallow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# まずはログに出ていたテスト対象ファイルを1つ使う
input_file = os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_1/102_1_1_full.json")

with open(input_file, "r", encoding="utf-8") as f:
    datas = json.load(f)

# 最初の1事例だけ見る
data = datas[0]

llm = Llama_Swallow()

print("===== 対話要約を生成中 =====")
dialogue_summary = llm.generate_summary(data["dialogue"])

print("\n===== 対話要約 =====")
print(dialogue_summary)

print("\n===== 候補アイテムごとの推薦理由 =====")

# 最初の3候補だけ表示
positive_indices = [i for i, s in enumerate(data["score"]) if s == 1]
negative_indices = [i for i, s in enumerate(data["score"]) if s == 0]

target_indices = positive_indices[:1] + negative_indices[:2]

for i in target_indices:
    candidate = data["candidates"][i]
    candidate_info = candidate["Summary"] + candidate["Feature"]
    gold_score = data["score"][i]

    print("\n" + "=" * 80)
    print(f"候補 {i+1}")
    print(f"正解ラベル: {gold_score}")

    if "Name" in candidate:
        print(f"観光地名: {candidate['Name']}")
    elif "name" in candidate:
        print(f"観光地名: {candidate['name']}")

    print("\n--- 観光地情報 ---")
    print(candidate_info[:500])

    print("\n--- 生成された推薦理由 ---")
    reason = llm.generate_rec_reason_with_dialogue(
        dialogue_summary,
        candidate_info,
        temperature=0.1
    )
    print(reason)
