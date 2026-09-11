# Direct Preference Optimizationを用いた現実的な対話型推薦のためのテキスト生成の洗練

日本語 | [English](README.md)

本リポジトリは、Direct Preference Optimization（DPO）を用いて「対話要約」と「アイテム推薦情報」の生成を改善する、対話型推薦システム（CRS）の研究実装です。

![提案手法の概要](images/proposal_flow.png)

[高解像度の図を見る](images/proposal_flow.pdf)

## 研究の背景

対話型推薦では、短いやり取りから性急に推薦してしまうことや、長い対話に含まれる暗黙的な嗜好を十分に統合できないことが課題になります。本研究ではSumRecのパイプラインを拡張し、候補アイテムを順位付けする前に、生成される対話要約とアイテム推薦情報をDPOで最適化します。

本実装は、次の実験工程を扱います。

1. 対話型推薦データの前処理
2. 教師あり学習用データと選好データの構築
3. DeBERTa／ModernBERTによるスコア予測器の学習
4. DPOによるテキスト生成モデルの学習
5. 推薦結果の生成
6. ランキング性能と生成テキスト品質の評価

## 提案手法

学習は2段階で行います。

- **スコア予測**：対話要約、アイテム推薦情報、候補情報から、DeBERTa／ModernBERTが候補アイテムのスコアを予測します。
- **選好最適化**：スコア予測器を使って選好ペアを構築し、DPOによって対話要約生成器とアイテム推薦情報生成器を改善します。

主な実験設定は次のとおりです。

- **生成モデル**：[Llama-3.1-Swallow-8B-v0.1](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-v0.1)
- **スコア予測器**：[DeBERTa-v3-japanese-large](https://huggingface.co/globis-university/deberta-v3-japanese-large)および[llm-jp-modernbert-base](https://huggingface.co/llm-jp/llm-jp-modernbert-base)
- **最適化**：Direct Preference Optimization、Optunaによるハイパーパラメータ探索
- **ランキング評価**：HR@k、MRR@k
- **生成文評価**：文字数、Distinct-1/2、BLEU、ROUGE

## 実験で確認した傾向

- Tabidachiコーパスでは、評価したベースラインに対してHR@1/3/5とMRR@1/3/5が改善しました。
- ChatRecでは、評価した手法の中で最も高いMRRを記録しました。
- アブレーションでは、対話要約生成器に対するDPO学習が特に重要であることを確認しました。

本リポジトリには評価コードと一部の学習メタデータを収録しています。データセットと学習済みモデルの重みは含みません。

## リポジトリ構成

```text
.
├── src/
│   ├── Tabidachi/              # Tabidachiの前処理・学習・生成・評価
│   └── ChatRec/                # ChatRecの前処理・学習・生成・評価
├── artifacts/
│   ├── models/
│   │   ├── deberta/            # tokenizer・設定・trainer state
│   │   └── modernbert/         # tokenizer・設定・trainer state
│   ├── dpo/recommendation/     # DPO adapter設定・trainer state
│   └── generated_text/         # 中間テキストの生成例
├── images/                     # 提案手法の図
├── eval_reason_overlap.py      # 生成理由の重複分析
├── eval_reason_quality.py      # 生成理由の品質分析
├── metrics_sentence.py         # 生成文の評価指標
└── requirements.txt            # 実験環境の依存関係
```

`artifacts/`は実験時の生成物を保存する領域です。実装コードである`src/`と分離しており、学習済みモデルの重みは含まれていません。実験メタデータとして、一部のtokenizerと設定ファイルを保持しています。

## データセット

データセット本体は収録していません。各提供元から取得し、それぞれの利用条件に従ってください。

### Tabidachiコーパス

- 提供元：[NII情報学研究データリポジトリ](https://www.nii.ac.jp/dsc/idr/rdata/Tabidachi/)
- 内容：旅行代理店の推薦対話
- 配置先：`data/Tabidachi/annotation_data/`

```text
data/Tabidachi/annotation_data/
├── annotations/
├── spot_info.json
└── タグ一覧.docx
```

### ChatRec

- 出典：[Ryutaro-A/SumRec](https://github.com/Ryutaro-A/SumRec)
- 内容：比較実験に使用する複数カテゴリの推薦対話
- 配置先：`data/ChatRec/chat_and_rec/`

## 実行環境

実験ではPython 3.10.12、PyTorch 2.4.1、Transformers 4.46.2、TRL 0.12.1、Optuna 4.1.0、およびNVIDIA A100 80 GBを4基使用しました。前処理や一部の評価は小規模な環境でも実行できますが、モデル学習の再現には大きなGPUメモリと実行時間が必要です。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

一部の学習スクリプトはWeights & Biasesを使用します。実行時にローカル環境で認証してください。認証情報は本リポジトリに保存していません。

## 再現手順

複数のスクリプトが実行位置からの相対パスを使用するため、対象コーパスのディレクトリに移動して実行します。

### Tabidachi

```bash
cd src/Tabidachi

python data_preprocessing.py
python create_dataset_1.py
python create_dataset_2.py
python create_dataset_3.py
python create_dataset_4.py

python train_deberta.py --method 'proposal&baseline1'
python dpo_summary_llm.py
python dpo_recommendation_llm.py
python create_recommend_data_proposal.py
python evaluate_from_recommend_data.py --method proposal
```

### ChatRec

```bash
cd src/ChatRec

python data_preprocessing.py
python create_dataset_1.py
python create_dataset_2.py
python create_dataset_3.py
python create_dataset_4.py

python train_deberta.py --method 'proposal&baseline1'
python dpo_summary_llm.py
python dpo_recommendation_llm.py
python create_recommend_data_proposal.py
python evaluate_from_recommend_data.py --method proposal
```

各コーパスのディレクトリには、別モデル、アブレーション、プロンプト別実験、事前計算済みランキング評価のスクリプトも含まれます。

## 公開範囲と利用条件

本リポジトリは、技術内容を確認できるよう研究コードと一部のメタデータを公開するものです。Tabidachi／ChatRecのデータと学習済みの重みは再配布しません。現時点では本リポジトリ独自のオープンソースライセンスを付与していません。保持しているtokenizer／設定ファイルと事前学習済みモデルの利用には、DeBERTa-v3-japanese-largeのCC BY-SA 4.0、llm-jp-modernbert-baseのApache License 2.0、Llama-3.1-SwallowのMeta Llama 3.1 Community Licenseを含む、各提供元の利用条件が適用されます。
