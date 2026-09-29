# Repository audit and publication changes

Audit base: `1808be5` (the only commit in the public history at inspection).
The repository was public and contained 244 tracked files: 86 Python files,
110 JSON files, 27 TXT files, 10 Markdown files, 8 tokenizer models, one PNG,
one PDF and .gitignore. No notebooks, shell scripts, tests or CI were tracked.
Source code occupied 1,124,864 bytes in the Windows checkout; Git artifact blobs occupied 298,002,408 bytes.

## Classification

| Classification | Files / directories | Decision |
|---|---|---|
| KEEP | All original Python source, both corpora, prompt variants, images | Preserve active and comparison paths; external research jobs cannot be inspected |
| KEEP | README.md / README_JP.md | Rewrite around the confirmed current reason + ModernBERT research |
| REORGANIZE | requirements.txt | Original preserved byte-for-byte in requirements/legacy-freeze.txt; root selects a focused profile |
| REORGANIZE | Root reason-analysis tools | Keep filenames, add input CLI and import guards; resolve existing artifacts correctly |
| REORGANIZE | Ranking metric code | Share CPU implementation between precomputed and ModernBERT evaluation |
| ARCHIVE | trainer_state.json, config.json, adapter_config.json, model cards | Retain identifiable experimental provenance in existing artifacts/ |
| DELETE | 30 checkpoint-local duplicate files | Identical copy remains in the same experiment directory; SHA-256 manifest records every path |
| DELETE CANDIDATE | Remaining tokenizers and generated reason histories | Confirm external backup and ongoing uses; do not assume AIX/local copies exist |

[Artifact inventory](artifact-inventory.csv) lists all original artifact files,
hashes, classification and rationale. [Deletion manifest](removed-duplicates.json)
records exact retained copies. [Source inventory](source-inventory.md) maps all
original Python files to imports, definitions and selected path literals.

## Before / after

Existing src/Tabidachi, src/ChatRec, root utilities, artifacts and images retain
their locations. Added docs/, examples/, results/, tests/, scripts/, requirements/,
data/README.md, .env.example and a CPU CI workflow. No original source file moved
or disappeared. Thirty redundant artifact files were removed, saving 113,925,569
bytes (about 108.6 MiB) from a checkout. Git history was not rewritten, so original
blobs remain recoverable and clone-history size is not reduced.

## Findings and changes

1. **Evaluation correctness:** ModernBERT's NDCG was unnormalized DCG; its Recall
   was actually HR. The shared evaluator now separates all four metrics. Tests
   cover multiple positives, no positives, cutoffs, ties and source grouping.
   Recompute historical metrics from saved scores; training behavior is unchanged.
2. **Runnable public entry:** a fully synthetic, dependency-free CPU example and
   JSON expected results. Empty/invalid inputs return errors instead of crashing
   with an index error or reporting an empty evaluation. Bare output filenames work.
3. **Import side effects:** Tabidachi create_dataset_1 and root reason tools no
   longer create/read data when imported. ModernBERT evaluation loads ML libraries
   only after parsing arguments, enabling CPU help and metric testing.
4. **Environment:** the original freeze contains mutually incompatible NumPy
   constraints through TensorFlow/JAX and predates ModernBERT. Preserve it as
   evidence, introduce separate profiles, add missing PEFT/text dependencies,
   and remove automatic pip installation from metrics_sentence.py.
5. **Execution order:** correct preference-generation dependence on a trained
   scorer. Document v16 CLI inputs/outputs, legacy paths and five-model expectations.
6. **Publication:** align both READMEs with actual code and the maintainer's stated
   contribution; remove unsupported performance claims; distinguish response
   generation as a future goal. Add provider and prior-work links.
7. **Ignore rules:** replace blanket source/data-format/results exclusions with
   known corpus, model, cache and output paths. Configs, CSV summaries, synthetic
   JSON examples, documentation and .env.example remain trackable.

## Security and data review

The inspected snapshot and current textual source/configuration were checked for
common GitHub/Hugging Face/API token formats, private-key markers, quoted credential
assignments and user-specific absolute paths. No matching secret was identified.
This is a scoped pattern review, not proof that arbitrary secrets or personal data
are absent. The public history contains only one snapshot; no external AIX files,
credentials or home-directory data were accessed. /tmp/wandb is an OS-specific
runtime path, documented rather than treated as a credential.

Generated-text files and inline example strings can contain corpus-derived
material. Their publication rights cannot be established by code inspection.
Keep them as review candidates and obtain the experiment owner's confirmation;
do not present them as newly cleared public examples. Existing artifact model
cards, model metadata and tokenizer licenses still need ownership/license review.
No license was added, no repository visibility changed, and no remote history purged.

## Verification and limitations

The local CPU suite passes 13 tests on Python 3.12.14. It covers exact expected
example output through a subprocess from another working directory, invalid inputs,
metric semantics and both corpora's preprocessing transformations. Syntax checks
cover all source files; imports are exercised only for safe CPU paths because
several research imports allocate models or read private data. Local documentation
links and every retained duplicate hash are checked by scripts/check_repository.py.
CI is configured for Python 3.10 and 3.12; its actual run result must be checked
on the pull request. In addition, 32 documented script commands/options, 20 ignore
policy cases and Python 3.10 syntax compatibility were checked. Both main
dependency profiles resolved in a pip dry-run on Windows / Python 3.12.14;
[validation.json](validation.json) records the limited check and profile hashes.
GPU training, pretrained-weight loading and licensed-corpus
processing are not claimed tested. See the cleanup report / PR for final checks.

## Remaining work that needs research context

- Improve and evaluate recommendation/non-recommendation reason quality, the active research task.
- Recover an environment lock per experiment; validate DPO API compatibility and ModernBERT on the research GPU.
- Confirm legacy Tabidachi→ChatRec summary-DPO paths and partial machine-specific splits.
- Preserve original data split manifests; avoid candidate/conversation leakage and stale copied split files.
- Supply complete weights for intended inference, and document 1..5 run provenance.
- Recompute corrected metrics; publish reproducible aggregate comparisons with SumRec and SumRec + DPO.
- Confirm AIX/local backup and external path use before deleting unique artifact candidates.
- Review corpus-derived text and third-party licenses before promoting the repository as a portfolio.
- Implement and evaluate response generation; it remains a stated goal, not a completed feature.

Artifact inventory sizes and SHA-256 values use Git blob bytes. For text files,
normalize Windows CRLF to Git LF before verifying a working copy. The deletion
check compared original and retained Git blobs directly; no content-only assumption
was used to choose duplicates.
