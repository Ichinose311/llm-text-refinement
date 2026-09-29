# Results and provenance

`example-metrics.json` is the output of the **synthetic CPU smoke test**, not an
experimental result. It contains two groups, six candidates, and no model calls.

No traceable HR/MRR benchmark table or held-out prediction files are available
in this checkout. Existing trainer_state.json files record training/validation
losses; these are not evidence of recommendation quality or baseline superiority.
The former README's improvement claims were removed pending a verifiable source.

For each future result, record: code commit, corpus/version/access conditions,
split manifest and grouping key, model revision, prompt file/hash, training seed,
hyperparameters, candidate count, checkpoint location, evaluator command/version,
tie policy, zero-positive policy, number of evaluated groups and missing runs.
Keep only reviewed aggregate results here; store raw predictions under data/ or
results/local/. Report mean and variation over seeds when those runs exist.

## Metric definitions

- HR@k: fraction of groups with at least one positive in the top k.
- Recall@k: fraction of each group's positives retrieved in the top k, averaged over groups.
- MRR@k: reciprocal rank of the first positive within k, zero if absent.
- NDCG@k: DCG divided by the ideal DCG at k, then averaged over groups.

The shared v16 evaluator uses linear relevance gain, descending predicted_score,
stable input-order ties, and (source_file, data_id) groups. Zero-positive groups
contribute zero. These policies are included in exported metrics reports.

## Evaluation correction

Before this cleanup, evaluate_modernbert_4input_v16.py reported unnormalized DCG
as NDCG and Hit Rate as Recall. It now uses the same tested metrics as the
precomputed evaluator, which also exports HR explicitly. Re-evaluate saved scores;
old numbers are not directly comparable. The 3-input ablation imports this helper
and receives the correction too. Training and predictions are unchanged.

Other historical evaluators are retained. Several label Hit Rate as Recall,
which agrees only for single-positive groups; use the canonical precomputed
evaluator on flat scored rows when making new comparisons. Legacy corpus
evaluators use numpy argsort ties and nested data formats, so identify the exact
evaluator in every report. Never tune fusion weights on the final test set.
