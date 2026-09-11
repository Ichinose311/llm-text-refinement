import train_modernbert_pairwise_4input_v16 as base

def build_text(x):
    return (
        "ユーザ要約:\n"
        f"{x.get('dialogue_summary', '')}\n\n"
        "観光地情報:\n"
        f"{x.get('candidate_information', '')}\n\n"
        "既存推薦文:\n"
        f"{x.get('item_recommendation_sentence', '')}"
    )

base.build_text = build_text

if __name__ == "__main__":
    base.main()
