import os
import json
import glob
from oss_llm_v16 import Llama_Swallow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_dir = os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_1/")

# 確認する事例数
MAX_EXAMPLES = 6

# 各事例で見る負例数
NEG_PER_CASE = 2

# 何ファイルまで探索するか
FILE_LIMIT = 80

# テストIDに絞るならここを使う
TEST_ID = {102, 114, 119, 203, 209, 305, 312, 316}

files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

# テストIDのファイルを優先
target_files = []
for f in files:
    name = os.path.basename(f)
    try:
        uid = int(name[:3])
    except ValueError:
        continue
    if uid in TEST_ID:
        target_files.append(f)

target_files = target_files[:FILE_LIMIT]

llm = Llama_Swallow()

seen_dialogues = set()
example_count = 0

for file_path in target_files:
    if example_count >= MAX_EXAMPLES:
        break

    filename = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        datas = json.load(f)

    # 1ファイルから多く取りすぎないように、先頭から順に1件だけ採用
    for data_id, data in enumerate(datas):
        if example_count >= MAX_EXAMPLES:
            break

        dialogue_key = json.dumps(data.get("dialogue", ""), ensure_ascii=False)[:500]

        # ほぼ同じ対話をスキップ
        if dialogue_key in seen_dialogues:
            continue
        seen_dialogues.add(dialogue_key)

        positive_indices = [i for i, s in enumerate(data["score"]) if s == 1]
        negative_indices = [i for i, s in enumerate(data["score"]) if s == 0]

        if not positive_indices or len(negative_indices) < 1:
            continue

        # 負例は先頭だけでなく、少し離れた候補も見る
        selected_negs = []
        if negative_indices:
            selected_negs.append(negative_indices[0])
        if len(negative_indices) >= 3:
            selected_negs.append(negative_indices[len(negative_indices)//2])
        elif len(negative_indices) >= 2:
            selected_negs.append(negative_indices[1])

        selected_negs = selected_negs[:NEG_PER_CASE]

        target_indices = positive_indices[:1] + selected_negs

        example_count += 1

        print("\n" + "#" * 100)
        print(f"事例 {example_count}")
        print(f"ファイル: {filename}")
        print(f"データ番号: {data_id}")
        print("#" * 100)

        print("===== 対話要約を生成中 =====")
        dialogue_summary = llm.generate_summary(data["dialogue"])

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

        # 1ファイルからは1件だけ
        break

print("\n" + "#" * 100)
print(f"完了: {example_count}事例を出力しました")
print("#" * 100)
