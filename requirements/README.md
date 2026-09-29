# Environment profiles

The CPU ranking example and unit tests use only the Python standard library.
Use Python 3.10+ for those commands. Local verification used Python 3.12.14;
CI is configured for 3.10 and 3.12.

| File | Purpose | Verification boundary |
|---|---|---|
| `legacy-core.txt` (root requirements.txt) | DeBERTa / Swallow / legacy DPO direct imports | Installation profile, not evidence that GPU experiments ran |
| `modernbert.txt` | ModernBERT scorer in a separate environment | Transformers 5.13.0 matches retained configs; full runtime not verified |
| `text-analysis.txt` | Optional Excel, BLEU, ROUGE, Japanese tokenization | Requires the locally held workbook and matching columns |
| `legacy-freeze.txt` | Exact original requirements.txt, preserved unchanged | Historical evidence; **do not use as a working lock** |

The original freeze pinned NumPy 1.23.5 with SciPy 1.14.1 (which requires
NumPy >=1.23.5,<2.3 but also has platform/Python constraints) and JAX 0.4.30
(which requires NumPy >=1.24), while TensorFlow 2.12.0 requires NumPy <1.24.
The JAX/TensorFlow constraints cannot both be satisfied. The core profile omits
unused TensorFlow/JAX, adds explicitly imported PEFT and tokenizer dependencies,
and uses NumPy 1.26.4. It removes frozen transitive CUDA wheels; select the PyTorch
build appropriate to the training machine rather than installing that old list.

The old Transformers 4.46.2 pin does not support ModernBERT, introduced in
[Transformers 4.48](https://huggingface.co/blog/modernbert). Retained ModernBERT
configs explicitly record 5.13.0, so installing the old freeze cannot reproduce
those runs. Legacy TRL APIs and the newer Transformers stack are kept separate.

No profile is claimed to reproduce a past run exactly. After validating the
chosen pipeline on the research server, save Python/CUDA/GPU versions, pip freeze,
model revision, seed, prompt version and dataset split with that run. Do not
upgrade the active AIX environment in place to try these profiles.

Both legacy-core and modernbert profiles passed `pip install --dry-run
--ignore-installed --only-binary=:all:` resolution on Windows / Python 3.12.14
using PyPI. No packages were installed and no GPU operation was performed.
This does not verify Linux/CUDA compatibility or legacy Trainer API behavior.
Resolved direct versions and profile hashes are in [validation.json](../docs/validation.json).
