# Synthetic example

Every name, label, score and sentence here was written for the smoke test. No
corpus dialogue or model output is reproduced. Scores are illustrative, not a
trained model's predictions, and the example does not demonstrate model quality.

Illustrative input: “I would like a quiet indoor activity.”
Illustrative summary: “Prefers quiet indoor places.”
Illustrative reason: “A museum offers indoor exhibits; crowd levels are unknown.”
Candidates: museum, park, gallery. Assigned scores: 0.9, 0.6, 0.3.
The descending order is museum → park → gallery.

`ranking/synthetic.json` contains two groups with three candidates each. It uses
the flat row schema emitted by the v16 evaluators. `score` is the relevance label;
`predicted_score` is the value used to sort candidates. The first group has two
positives, intentionally distinguishing Hit Rate from Recall.

Run from the repository root:

```bash
python src/Tabidachi/evaluate_precomputed_ranking.py --input-dir examples/ranking --ks 1 3 5
```

The [expected report](../results/example-metrics.json) is checked by the CLI test.
The original `evaluate_from_recommend_data.py` files use different nested formats;
do not pass their output directly to this flat-row evaluator.
