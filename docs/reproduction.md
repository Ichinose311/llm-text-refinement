# Reproduction guide

Run commands from the repository root unless a block explicitly changes directory.
The active research line is **v16 recommendation/non-recommendation reasons +
four-input ModernBERT**. Reason quality is under development. Legacy DPO scripts
are preserved for comparison and ongoing experiments.

## 1. What is reproducible without research infrastructure

Python 3.10+ is sufficient; no pip install is needed:

```bash
python src/Tabidachi/evaluate_precomputed_ranking.py --input-dir examples/ranking --ks 1 3 5 --save-json outputs/example-metrics.json
python -m unittest discover -s tests -v
python scripts/check_repository.py
python -m compileall -q src scripts tests eval_reason_overlap.py eval_reason_quality.py metrics_sentence.py
```

These commands check evaluation, preprocessing transformations, syntax and local
references. They do not verify training, LLM generation or recommendation quality.

## 2. Research environment and data

Create a dedicated environment using [requirements profiles](../requirements/README.md).
The original experiment environment is not reproducible from its freeze unchanged.
Keep the current server environment intact while testing these profiles separately.
Model IDs in the code download pretrained weights from Hugging Face; no trained
scorer, DPO model or LoRA weights are distributed in this repository.

Use [data placement instructions](../data/README.md). The legacy scripts often
resolve datasets relative to `__file__`, but checkpoint names relative to the
working directory. v16 CLI paths are relative to the working directory.
Some legacy files set HF_HOME themselves. `.env.example` documents variables;
there is no automatic dotenv loading. To avoid W&B reporting in a local check,
export `WANDB_MODE=disabled` (PowerShell: `$env:WANDB_MODE='disabled'`). Some legacy
scripts also send W&B alerts. No training script was run during this cleanup.

## 3. Preprocessing

**Tabidachi preprocessing deletes and rebuilds formatted_data.** Back up existing
local output before running this original script. The cleanup did not execute it
on real data. It expects annotations/*.json and spot_info.json, not a workbook.

```bash
python src/Tabidachi/data_preprocessing.py
python src/Tabidachi/create_dataset_1.py
```

The first command extracts dialogue context and candidate/mentioned items; the
second merges consecutive utterances and assigns binary candidate labels.

ChatRec preprocessing maps speakers to A/B and thresholds questionnaire scores at 3:

```bash
python src/ChatRec/data_preprocessing.py
python src/ChatRec/create_dataset_1.py
```

The second command currently shuffles without a fixed seed and copies into
Train/Test/Valid category folders. It can leave stale files in existing folders.
Use a fresh destination / saved split manifest when preparing a reproducible run;
do not rerun it over an active split. This historical split behavior is unchanged.

## 4. v16 reason generation

These are real CLI commands requiring the Swallow/PEFT environment, local data and
GPU memory. They were checked against argument definitions, **not run with weights**.
The first two commands below intentionally limit generation for a GPU smoke test.
Remove the limits only after validating the output and selecting the intended split.

```bash
python src/Tabidachi/create_dialogue_summary_v16.py --input-dir data/Tabidachi/datasets_1 --output data/Tabidachi/v16_summary_smoke --max-files 1 --limit 1
python src/Tabidachi/create_dataset_2_clean_v16.py --input-dir data/Tabidachi/datasets_1 --summary-dir data/Tabidachi/v16_summary_smoke --output data/Tabidachi/v16_reason_smoke --max-files 1 --limit 1 --max-candidates 2
python src/Tabidachi/create_dataset_4input_v16_all.py --reason-data-dir data/Tabidachi/v16_reason_smoke --output data/Tabidachi/v16_four_input_smoke --max-files 1 --max-rows 2
```

Do not train or report benchmark quality on these two smoke-test rows. The output
retains source_file/data_id/candidate_index so subsequent full datasets can be
partitioned and joined without losing candidate identity.

## 5. Training the current ModernBERT scorer

In the **separate ModernBERT environment**, prepare full four-input training data
under `data/Tabidachi/v16_four_input_train/` and held-out data under
`data/Tabidachi/v16_four_input_test/`. These names denote locally prepared inputs,
not files supplied by this repository. Generate their four text fields as above
using saved corpus splits and without the smoke limits. Preserve the original
Tabidachi held-out IDs when comparing against existing baselines.

```bash
python src/Tabidachi/train_modernbert_4input_v16.py --data-dir data/Tabidachi/v16_four_input_train --output-dir outputs/modernbert_regression --epochs 3 --seed 42 --bf16
python src/Tabidachi/train_modernbert_pairwise_4input_v16.py --data-dir data/Tabidachi/v16_four_input_train --output-dir outputs/modernbert_pairwise --epochs 3 --seed 42 --bf16
```

`--bf16` requires compatible hardware; omit it if unsupported. Default max length
is 8192, so choose batch size / accumulation for available GPU memory. These two
commands train alternative scorers; running both is not required for inference.
Regression splits shuffled candidate rows for internal validation, while pairwise
training splits `(source_file, data_id)` groups. Neither internal split substitutes
for a separately held-out conversation-level test set. No split was changed here.

## 6. Inference and evaluation

After training a complete ModernBERT model, generate scores for held-out rows:

```bash
python src/Tabidachi/evaluate_modernbert_4input_v16.py --input-dir data/Tabidachi/v16_four_input_test --evaluator-dir outputs/modernbert_pairwise --ks 1 3 5 --save-scored outputs/scored/heldout.json
python src/Tabidachi/evaluate_precomputed_ranking.py --input-dir outputs/scored --ks 1 3 5 --save-json results/local/heldout-metrics.json
```

Keep **only scored-row lists** in outputs/scored; the second command consumes all
*.json there. The metric report belongs outside that directory. The input model
must include weights, config and tokenizer; artifacts/ contains metadata only.
This is batch ranking inference, not a conversational response-generation API.
Response generation is a future research goal and is not claimed implemented.

The canonical evaluator groups by `(source_file, data_id)`, averages groups equally,
keeps input order for equal predictions, and includes zero-positive groups as zero.
See [metric corrections](../results/README.md) before reusing old ModernBERT results.

## 7. Optional v16 reason DPO experiments

Use the Swallow/PEFT environment again. First train a DeBERTa reason scorer on
full, independently selected reason rows, then build preference pairs using it:

```bash
python src/Tabidachi/train_deberta_reason_v16.py --data-dir data/Tabidachi/v16_reason_train --output-dir outputs/deberta_reason --epochs 1 --seed 42
python src/Tabidachi/create_dataset_4_reason_v16.py --input-dir data/Tabidachi/datasets_1_train --summary-dir data/Tabidachi/v16_summary_train --evaluator-dir outputs/deberta_reason --output data/Tabidachi/v16_preferences_train
python src/Tabidachi/dpo_recommendation_llm_reason_v16.py --data-dir data/Tabidachi/v16_preferences_train --output-dir outputs/reason_dpo --epochs 1
python src/Tabidachi/create_recommend_data_reason_v16_dpo.py --input-dir data/Tabidachi/datasets_1_test --summary-dir data/Tabidachi/v16_summary_test --adapter-dir outputs/reason_dpo --output data/Tabidachi/v16_reason_dpo_test
```

The *_train/*_test paths must be prepared from the saved split, with matching
source_file/data_id summaries. They are not produced automatically by the legacy
preprocessor. Generated DPO reason rows can be expanded to four inputs using
create_dataset_4input_v16_all.py and scored by the chosen compatible scorer.
Record whether a scorer saw baseline reasons or DPO reasons during training.

## 8. Legacy SumRec-style / summary-and-recommendation DPO path

Dependency order (applies to both corpora):

| Stage | Input | Output / dependency |
|---|---|---|
| data_preprocessing | provider dataset | formatted_data or processed_data |
| create_dataset_1 | preprocessed corpus | dialogue/candidates/labels or corpus split |
| create_dataset_2 | datasets_1 + Swallow | scorer examples in datasets_2 |
| train_deberta | datasets_2 | trained DeBERTa scorer |
| create_dataset_3 / 4 | datasets_1 / 2 + trained scorer | summary / recommendation preferences |
| dpo_summary_llm / dpo_recommendation_llm | preferences | DPO generators |
| create_recommend_data_* | generators + scorer + test data | nested recommendation JSON |
| evaluate_from_recommend_data | nested JSON | HR@1..10, MRR@1..10 |

The following legacy commands exist, but **review the blockers below before
executing the whole sequence**. Run inside src/Tabidachi or src/ChatRec so saved
checkpoint names resolve correctly:

```bash
cd src/Tabidachi
python create_dataset_2.py
python train_deberta.py --method 'proposal&baseline1'
python create_dataset_3.py
python create_dataset_4.py
python dpo_summary_llm.py
python dpo_recommendation_llm.py
python create_recommend_data_proposal.py
python evaluate_from_recommend_data.py --method proposal
```

For baseline2, train with `--method baseline2`, then run
`python create_recommend_data_baseline2.py` and
`python evaluate_from_recommend_data.py --method baseline2` in that corpus directory.
Baseline1 uses the proposal&baseline1 scorer and create_recommend_data_baseline1.py.

Known blockers retained for experiment-owner review:

- Tabidachi dpo_summary_llm.py reads ChatRec datasets_3, including both active
  training and validation paths. This may reflect transfer/retraining intent;
  it has not been silently rewritten as a Tabidachi run.
- Tabidachi create_dataset_3.py currently selects only TRAIN_VALID_ID_CLOVA.
  Check the full split and machine partitions before generating all preferences.
- DPO scripts save an initial *_results_1 model, while proposal generation loops
  over models 1..5. The *_more scripts require experiment-specific alignment.
  Missing model folders cannot be replaced with metadata-only artifacts.
- Old DPOTrainer keyword usage must be checked against the installed TRL version.
  Installing a dependency profile is not evidence that all historical APIs work.
- Legacy evaluators may skip missing runs and still print completion; report the
  number of evaluated models instead of implying all five were present.

## 9. Optional text diagnostics

The root heuristics now accept explicit paths and do not run on import:

```bash
python eval_reason_quality.py --input artifacts/generated_text/reason_examples/reason_examples_v11.txt
python eval_reason_overlap.py --input artifacts/generated_text/reason_examples/reason_examples_v16_multi.txt
```

These files were already in the public snapshot. They are retained pending review,
not endorsed as safe examples for redistribution. `metrics_sentence.py` additionally
needs requirements/text-analysis.txt and the local cloud_worker_tabidachi_datasets.xlsx
workbook with the exact columns named in the script. It no longer installs packages
automatically. Its BLEU/ROUGE tokenization, especially Japanese ROUGE behavior, needs
validation before reporting text-quality comparisons.
