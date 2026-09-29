# Method and contribution map

## Current research and confirmed contribution

The maintainer confirmed that the active line is recommendation/non-recommendation
reason generation plus four-input ModernBERT. Reasons are still being improved.
Their contribution is explicitly linking user preferences from the dialogue summary
to item features, generating grounds for and against recommendation, and adding
those grounds to SumRec as a fourth scorer input. The three-input SumRec structure
is inherited prior work. Exceeding SumRec and SumRec + DPO and extending to response
generation are goals; no verified response-generation pipeline or superiority
result is claimed here.

## What is implemented

The legacy path is visible in each corpus's `create_dataset_2.py`,
`train_deberta.py`, `create_dataset_3.py`, `create_dataset_4.py`, DPO scripts and
`create_recommend_data_proposal.py`. The scorer input is user summary, item
recommendation text and candidate information, separated by tokenizer separators.
`baseline2` omits the recommendation text; `baseline1` adds it without generator
DPO. Tabidachi ablation1 uses a trained summary generator with an untrained
recommendation generator; ablation2 reverses this.

In legacy preference construction, the scorer compares generated alternatives.
For a positive candidate, a higher predicted relevance selects a preferred text;
the generation loops currently select positive candidates. The negative branch
exists in helper functions but should not be mistaken for proof that negative
training examples were used. Summary and recommendation DPO are separate stages.

## v16 extension

`oss_llm_v16.py` generates reasons for and against recommendation conditioned on
the summary and candidate information. `create_dataset_2_clean_v16.py` joins
stored summaries with candidates, maintaining source_file, data_id and candidate_index.
`create_dataset_4input_v16_all.py` adds an item recommendation text to reason rows.

The four fields consumed by the ModernBERT scorers are:

1. dialogue_summary;
2. candidate_information;
3. item_recommendation_sentence;
4. v16_reason_sentence.

The regression variant learns a single relevance score with MSE. The pairwise
variant forms positive/negative pairs within a dialogue group and minimizes
softplus(margin - (positive_score - negative_score)). The three-input ablation
omits the personalized-reason input; it is a separate experimental script, not the default model.

For reason DPO, `create_dataset_4_reason_v16.py` samples v16 outputs and adds
generic reason variants, then chooses the reason whose predicted score is
closest to the gold label and rejects a distinct distant alternative.
`dpo_recommendation_llm_reason_v16.py` converts them to prompt/chosen/rejected,
skips identical pairs and trains a LoRA adapter. These details differ from the
legacy highest-score preference rule and must be reported when comparing runs.

## Evaluation boundaries

The scorer is a learned proxy, not a factuality oracle. Heuristic word checks and
overlap counts in the root utilities do not prove hallucination detection.
Avoid training/evaluating the preference generator on the same held-out targets.
Four-input regression currently shuffles candidate rows before validation split,
so candidates from one dialogue can cross that boundary. Pairwise training splits
dialogue groups, but different turns from one conversation may still cross it.
Keep a corpus-appropriate conversation-level held-out test split for claims.
No split behavior was silently changed by the publication cleanup.

## Reviewable implementation versus personal authorship

| Technical work visible in this repository | Review entry |
|---|---|
| Corpus adaptation and relevance labels | [Tabidachi transform](../src/Tabidachi/create_dataset_1.py), [ChatRec preprocessing](../src/ChatRec/data_preprocessing.py) |
| Preference-pair selection | [v16 pairs](../src/Tabidachi/create_dataset_4_reason_v16.py) |
| Generator optimization | [v16 DPO](../src/Tabidachi/dpo_recommendation_llm_reason_v16.py) |
| Personalized reason prompting | [v16 generator](../src/Tabidachi/oss_llm_v16.py) |
| Four-input representation and pairwise loss | [pairwise scorer](../src/Tabidachi/train_modernbert_pairwise_4input_v16.py) |
| Held-out score evaluation | [CPU evaluator](../src/Tabidachi/evaluate_precomputed_ranking.py), [tests](../tests/test_ranking.py) |

The public history at the audit base contains one snapshot commit; personal
attribution above is based on the maintainer’s explicit clarification. Authorship
of the remaining utilities and DPO scripts is not inferred from their presence.
For internship review, explain the reason-generation design decision and add
evidence-backed comparisons when available. Do not claim ownership of pretrained
models, DPO, ModernBERT or prior SumRec work.
