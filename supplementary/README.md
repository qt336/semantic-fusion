# Anonymous supplementary code

This package accompanies **Do Large Language Models Fuse Adjacent Tokens at
Lower Layers Based on Semantic Relatedness?** It contains runnable code,
installation instructions, and the fixed input data needed to repeat the
experiments. It does not contain experimental results, saved model responses,
scoring annotations, result tables, trained weights, or training logs.

All commands below are run from this directory. API keys are supplied in the
reviewer's own environment; no credentials are distributed.

## Contents

- `toy/`: synthetic data generation, standard Transformer and hybrid models,
  training, evaluation and implementation checks.
- `llm/`: literal prompt construction, input validation, API evaluation,
  first-proposed-answer scoring and independent tokenizer verification.
- `llm/data/items.jsonl`: the 3,000 fixed input items and their gold labels.
  Gold labels specify the expected second constituent, not a model response.
- `llm/data/candidate_pools.json`: the English and Chinese sampling pools.
- `llm/data/tokenizer_manifest.json`: pinned official tokenizer repositories,
  revisions and artifact checksums, used to download/validate dependencies.
- `checks/`: offline request/scoring regression tests using small test fixtures.
- `MANIFEST.sha256`: checksums of the distributed files.

## Installation

Use Python 3.10, the version used for the packaging checks. Toy training uses
POSIX process locking and supports Linux and other POSIX environments.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency pins use PyTorch 2.5.1 and NumPy 1.26.4. Install a PyTorch build
appropriate for the target GPU/CUDA environment. Full training uses CUDA by
default and does not silently fall back to CPU. Input validation and scoring
use only the Python standard library.

## Toy-model training

Both architectures use four pre-LayerNorm blocks, width 256, eight heads,
GELU MLP width 1,024, a final LayerNorm, a 336-token vocabulary, and interleaved
RoPE on Q/K with base 10,000. The hybrid uses normalized `ELU(x)+1` kernel
attention with causal depthwise Q/K/V convolutions of width three in layers
1 and 3; layers 2 and 4 retain softmax attention. It is not a full Kimi Delta
Attention implementation.

A full seeded dataset contains 270,000 arithmetic and 135,000 retrieval
training examples. The operator pairs `(0,1)`, `(0,2)` and `(2,1)` are excluded
from retrieval training, including the target-to-answer boundary. Arithmetic
is modulo 331. The dataset generator supports seeds 101--105; using the same
seed shares the dataset and common parameter initializations across the two
architectures.

Training uses 1,200 epochs, effective batch size 8,190, AdamW with betas
`(0.9,0.999)` and weight decay `0.1`, a 500-update warmup, and cosine decay from
`1e-3` to `1e-4`. Microbatches accumulate to the effective batch size. Model,
data and loss details are specified in the source and Appendices A and D.

Check the implementation and run short CPU training:

```bash
python toy/validate.py
python toy/train.py --architecture transformer --device cpu --small --epochs 1 --batch-size 256 --microbatch 128 --root outputs/smoke_transformer
python toy/train.py --architecture hybrid --device cpu --small --epochs 1 --batch-size 256 --microbatch 128 --root outputs/smoke_hybrid
```

Run the full protocol for one seed or all five seeds:

```bash
python toy/train.py --architecture transformer --seed 101 --device cuda
python toy/train.py --architecture hybrid --seed 101 --device cuda
bash toy/run_seeds.sh transformer
bash toy/run_seeds.sh hybrid
```

Independent seeds can run on different GPUs using `CUDA_VISIBLE_DEVICES` for
separate invocations. Each invocation uses one GPU. Repeating an invocation
with the same output directory and arguments resumes its checkpoint. Results
and checkpoints are generated under `outputs/`; they are not distributed.
The directed first-layer attention-edge evaluation is retained for the
Transformer. The hybrid run reports its unablated evaluation.

**Implementation provenance.** The available toy code was reconstructed from
the paper's specification and supports both architectures. The original
Transformer run files are not included. Retraining this implementation
repeats the specified protocol, rather than replaying the original random
examples or guaranteeing identical historical percentages. The hybrid
implementation corresponds to the reconstructed architecture in Appendix D.

## LLM inputs and tokenizer verification

Each of the six language/condition combinations has 500 input items. The
semantic sets contain 25 English expressions and 118 Chinese pairs. Comma
items are explicitly paired with their semantic counterparts. Random triples
are the actual independently sampled inputs, not shuffled semantic items;
the Chinese pool has 189 characters and the English pool has 2,694 filtered
WordNet entries. Use the fixed items to repeat the experiment on the same
questions.

```bash
python -m llm.validate_data
python -m pip install -r requirements-tokenizers.txt
python -m llm.verify_tokenizers --model deepseek --download --output outputs/tokenizers_deepseek.json
python -m llm.verify_tokenizers --model kimi --download --output outputs/tokenizers_kimi.json
python -m llm.verify_tokenizers --model hy4 --download --output outputs/tokenizers_hy4.json
```

The tokenizer checks compute fresh token IDs. For contiguous Chinese and
spaced English, each initial sequence must contain exactly three aligned
tokens. The complete question is longer. Comma sequences allow extra separator
tokens. The scripts verify text round trips and alignment at the start of the
actual prompt. No saved tokenization results are required.

Only the pinned tokenizer artifacts are downloaded, not model weights.
`--local-dir` accepts an existing copy instead. Artifact checksums are verified
before loading. Kimi requires the checksum-verified custom tokenizer code
from its official pinned revision. Fresh independent re-encoding could not be
completed during packaging because the tokenizer host was unreachable.

## Run the LLM experiment

Requests use a single user message, no system/history messages,
`temperature=0`, and `max_tokens=32`:

| CLI model | Requested model | Provider | Reasoning control | Key variable |
|---|---|---|---|---|
| `kimi` | `kimi-k3` | DashScope | `reasoning_effort="none"` | `KIMI_API_KEY` |
| `deepseek` | `deepseek-v4-pro` | DashScope | `enable_thinking=false` | `DEEPSEEK_API_KEY` |
| `hy4` | `hy4-preview` | Tencent MaaS | `thinking={"type":"disabled"}` | `HY4_API_KEY` |

Set the relevant key variable privately in your environment. Optional
`KIMI_BASE_URL`, `DEEPSEEK_BASE_URL` and `HY4_BASE_URL` override the public
provider endpoints. Keys are not written to outputs.

Inspect payloads and run offline regression tests without API requests:

```bash
python -m llm.run --model kimi --dry-run --limit-per-condition 2
python -m unittest discover -s checks -v
```

Run all six conditions for each model, making 3,000 requests per model:

```bash
python -m llm.run --model kimi
python -m llm.run --model deepseek
python -m llm.run --model hy4
```

Use `--language`, `--group`, and `--limit-per-condition` for a subset and
`--workers` to control concurrency. Output is flushed after each response;
repeating a command resumes without requesting completed items again. Only
transport failures and retryable HTTP errors are retried. Wrong answers are
retained. Returned reasoning is flagged as a protocol failure.

Score the responses generated by your own run:

```bash
python -m llm.score --responses outputs/responses/kimi.jsonl --output outputs/kimi_scores
python -m llm.score --responses outputs/responses/deepseek.jsonl --output outputs/deepseek_scores
python -m llm.score --responses outputs/responses/hy4.jsonl --output outputs/hy4_scores
```

The scorer writes JSON, CSV and Markdown tables. It requires an explicit
response file and does not depend on any results bundled with the code.
Failed requests and protocol failures are excluded from completed-response
counts; check completeness before comparing 500-item conditions.

The scoring rule ignores introductory answer phrases and complete question
or input echoes when identifying the first proposed answer. Later corrections
never override an earlier answer. For example, `The word is cream` is scored
by `cream`, while `ice. Actually, cream` is scored by `ice`. A bare compound
is scored by its first constituent. Previously unseen free-form responses may
need documented manual review under Appendix B.2. The extractor preserves the
rules used in the paper.

## Validation and anonymization

Both architectures completed short CPU training and scientific implementation
checks. The scorer was also checked against the existing study responses
outside this package. No new paid model generations or full GPU runs were
performed during packaging.

This package excludes personal paths, credentials, Git history, local
configuration, environments, diagnostics, and experimental output artifacts.
The WordNet data notice is retained in `THIRD_PARTY_NOTICES.md`.
