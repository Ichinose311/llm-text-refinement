# Retained experimental metadata

These folders are provenance, **not usable model checkpoints**: trained weights
are absent. No AIX or other local backup was assumed to exist.

- trainer_state.json, model config.json and adapter_config.json document run settings.
- Unique tokenizers and generated text are retained pending backup/publication review.
- Thirty checkpoint-local files identical byte-for-byte to files in the same
  experiment's parent directory were removed (113,925,569 bytes, about 108.6 MiB).
  Source imports and path literals do not reference these artifact directories.
  Every deletion has a retained copy and SHA-256 in the manifest below.

[Full classification](../docs/artifact-inventory.csv) ·
[Exact-copy deletion manifest](../docs/removed-duplicates.json)

If an external experiment requires an old checkpoint-local path, copy the manifest's
retained_copy back to path after verifying its SHA-256. Do not use these metadata-only
folders as substitutes for complete models. Git history still contains the original
blobs; reducing checkout size does not purge history or reduce existing clone size.

Generated text can contain corpus-derived material even though the raw datasets
are absent. Confirm redistribution rights before featuring any real dialogue or
generated excerpt in a portfolio; the public demo uses synthetic content only.

Artifact inventory sizes and SHA-256 values use Git blob bytes. For text files,
normalize Windows CRLF to Git LF before verifying a working copy. The deletion
check compared original and retained Git blobs directly; no content-only assumption
was used to choose duplicates.
