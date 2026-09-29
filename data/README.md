# Dataset placement

Raw or derived research data is not bundled. Keep local files in these ignored
corpus directories, which the legacy code resolves relative to its source file:

```text
data/
├── Tabidachi/
│   ├── annotation_data/annotations/*.json
│   ├── annotation_data/spot_info.json
│   ├── formatted_data/        # preprocessing output
│   ├── datasets_1/            # dialogue, candidates, relevance labels
│   ├── datasets_2/            # summary, candidate information, generated text, score
│   ├── datasets_3/            # summary preference pairs (legacy)
│   └── datasets_4/            # recommendation preference pairs (legacy)
└── ChatRec/
    ├── chat_and_rec/
    │   ├── except_for_travel/
    │   ├── travel/
    │   └── no_restriction/
    ├── processed_data/
    └── datasets_1/            # Train-*, Test-*, Valid-* category directories
```

Tabidachi: obtain access through the [NII provider](https://www.nii.ac.jp/dsc/idr/rdata/Tabidachi/).
ChatRec: the [SumRec authors' repository](https://github.com/Ryutaro-A/SumRec)
provides `data/chat_and_rec/`. Obtain that directory from the provider and retain
its category subdirectories here. Confirm the dataset usage conditions; the
software license alone does not establish permission to redistribute dialogues.

Do not put licensed dialogue excerpts into examples/ or results/. The
[public example](../examples/README.md) is entirely synthetic.

The original Tabidachi preprocessing script deletes and recreates formatted_data
when run. Copy or back up local outputs before using it. This cleanup did not run
that script on real data or change its extraction/annotation semantics.
