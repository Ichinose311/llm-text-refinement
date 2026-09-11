import re
from pathlib import Path

path = Path("reason_examples_v11.txt")
text = path.read_text(encoding="utf-8")

rec_lines = re.findall(r"推薦する理由:\s*(.*)", text)
not_rec_lines = re.findall(r"推薦しない理由:\s*(.*)", text)

all_reason_lines = rec_lines + not_rec_lines

suspicious_words = [
    "ヨーカドー", "秋田県", "武家屋敷", "ヨーグルト", "チーズ",
    "温水プール", "サウナ", "岩盤浴", "観覧車", "遊園地",
    "動物園", "水族館", "テーマパーク", "解決策", "解答例",
    "ヒント", "データセット", "トレーニング", "観測地", "観察地"
]

generic_phrases = [
    "直接一致するとは限りません",
    "一部合う可能性があります",
    "対応は限定的です",
    "強く推薦する根拠は弱い",
    "明確な対応は限定的"
]

failure_phrases = [
    "生成に失敗",
    "出力されませんでした"
]

def count_contains(lines, words):
    return sum(any(w in line for w in words) for line in lines)

def avg_len(lines):
    if not lines:
        return 0
    return sum(len(line) for line in lines) / len(lines)

print("===== 生成品質評価 =====")
print(f"推薦する理由の件数: {len(rec_lines)}")
print(f"推薦しない理由の件数: {len(not_rec_lines)}")
print(f"全理由文の件数: {len(all_reason_lines)}")
print()

print("===== 形式 =====")
print(f"推薦する理由と推薦しない理由の件数差: {abs(len(rec_lines) - len(not_rec_lines))}")
print(f"平均文字数（推薦する理由）: {avg_len(rec_lines):.1f}")
print(f"平均文字数（推薦しない理由）: {avg_len(not_rec_lines):.1f}")
print()

print("===== 問題表現 =====")
print(f"入力外・形式崩れ疑い語を含む文数: {count_contains(all_reason_lines, suspicious_words)}")
print(f"生成失敗文数: {count_contains(all_reason_lines, failure_phrases)}")
print(f"汎用表現を含む文数: {count_contains(all_reason_lines, generic_phrases)}")
print()

print("===== ユニーク性 =====")
print(f"ユニークな推薦理由数: {len(set(rec_lines))}")
print(f"ユニークな推薦しない理由数: {len(set(not_rec_lines))}")
print()

print("===== 入力外・形式崩れ疑いの文 =====")
for line in all_reason_lines:
    if any(w in line for w in suspicious_words):
        print("-", line)

print()
print("===== 汎用表現の文 =====")
for line in all_reason_lines:
    if any(w in line for w in generic_phrases):
        print("-", line)
