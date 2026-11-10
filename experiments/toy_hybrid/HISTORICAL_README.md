# Reconstructed toy-model hybrid experiment

This project was rebuilt from Sections 4–5 and Appendix A of the Overleaf
paper on 2026-09-12. The author confirmed that the original training code is
unavailable. This is a new paired experiment, not an exact reproduction of
the paper's original random examples or reported numbers.

## Paired design

Run both architectures with seeds 101–105 (10 runs total). A seed determines
one saved dataset, identical shuffle order, and identical initial values for
every shared parameter. Each run uses the paper's 1,200 epochs, batch size
8,190, 500-update linear warmup, cosine decay from 1e-3 to 1e-4, AdamW
betas (0.9, 0.999), weight decay 0.1, and no dropout or gradient clipping.
Microbatches of at most 2,048 accumulate to the same global weighted loss
and optimizer batch; this does not change the batch size to 2,048.

Both models have 4 pre-LayerNorm blocks, width 256, 8 heads, GELU MLP width
1,024, interleaved RoPE (base 10,000), final LayerNorm, and vocabulary 336.
Weights use PyTorch defaults. Embedding/output weights are untied and linear
layers have bias. Training uses float32 with TF32 enabled, equally for both.

| Architecture | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Parameters |
|---|---|---|---|---|---:|
| Transformer | softmax | softmax | softmax | softmax | 3,331,920 |
| Hybrid | linear + causal convolution | softmax | linear + causal convolution | softmax | 3,338,064 |

The hybrid adds a depthwise causal convolution of kernel width 3 separately
to projected Q/K/V (implemented as one grouped convolution), before RoPE.
Its linear attention uses the positive feature map `ELU(x)+1` on rotated Q/K:
`sum_{j<=i} phi(q_i)^T phi(k_j) v_j / sum_{j<=i} phi(q_i)^T phi(k_j)`.
This is normalized kernel linear attention, inspired by
[Katharopoulos et al. (2020)](https://proceedings.mlr.press/v119/katharopoulos20a.html).
Because sequences are at most 12 tokens, the implementation evaluates its
small causal kernel matrix directly. A recurrent-equivalence check verifies
the same outputs. The model does not contain a delta-rule memory or Kimi's
full architecture. The comparison changes both attention and local convolution;
it does not independently attribute any effect to either component.

## Data and reconstruction choices

- 270,000 arithmetic examples: 90,000 distinct operand pairs per operation,
  disjoint from 4,096 held-out pairs per operation. Training operands/results
  cover all 331 values for each operation.
- 135,000 retrieval examples: distinct `(memory, target_index)` signatures,
  unique memory tokens, lengths 3–6, no designated adjacent operator pair,
  and no forbidden ordered `(query_target, answer)` pair.
- 2,000 standard retrieval examples follow those constraints and have no
  signature overlap with training.
- 7,680 semantic retrieval examples: 2,560 per designated pair. For each pair,
  640 examples use each memory length 3–6; the operator pair immediately
  precedes the queried token and its position is random. Other memory tokens
  are distinct values in 3–330. Evaluation examples may repeat, particularly
  at length 3. The paper specifies uniqueness for retrieval training only.
- Controls `a1b` and `a2b`: 2,560 distinct triples each, with `a,b` drawn from
  3–330. These controls do not contain any designated operator pair.

The paper does not fully specify operator-test placement sampling, control-set
sampling/counts, the random-generator implementation, initialization biases,
embedding tying, numeric precision, or the denominator of weighted CE.
Those choices are explicitly fixed here and shared between architectures.
The standard retrieval pool allows individual 0/1/2 tokens but excludes the
designated pairs, matching the stated retrieval-training exclusion rules.

Complete sequences include BOS, the task input, the answer, and EOS; they are
padded to 12 tokens and shifted for causal next-token training. Retrieval
weights are 0.1 for memory, 2 for query target, 10 for answer. Arithmetic
weights are 0.1 for operands and 10 for result. Other targets have weight 1;
PAD has weight 0. Loss is `sum(weight * CE) / sum(weight)` per optimizer batch.

## Evaluation and ablation

Accuracy uses the argmax at the input's final position, before any gold answer
is visible. Report arithmetic, standard retrieval, each operator pair, and
both controls. For semantic cases, Correct / First / EOS / Other partition
all predictions. Save individual predicted IDs for independent rechecking.

Also remove the directed attention edge from the second operator token to
the first in layer 0 and reevaluate arithmetic and semantic retrieval.
For hybrid layers this zeros only that kernel-weight entry and renormalizes;
the convolution remains active. It is an attention-edge intervention, not
removal of every possible path between the two tokens.

## Persistence and resources

The `qt1` screen session runs `launch.py`. It scans all GPUs and starts one
seed run on each idle GPU that passes a real CUDA matrix-multiplication probe.
Busy GPUs and failed devices are not reset or preempted. It checks again as
devices become available. CUDA initialization failures leave runs queued,
with **no result**; CPU smoke checks are saved separately under `smoke/`.

At setup, GPUs 0–2 were idle, 3 was not accessible through NVML, and 4–7 were
occupied by another account. New CUDA contexts failed with error 999, even
when GPU 0 was isolated in a temporary container. Training cannot begin until
the host allows new CUDA contexts. No driver reset or reboot is attempted.
The native driver probe subsequently showed `/dev/nvidia-uvm` failing to open
with `EIO`, and GPU 0 reported `GPU Recovery Action: Reboot`. NVIDIA documents
that recovery state as requiring an OS reboot:
https://docs.nvidia.com/deploy/nvidia-smi/ . GPU 3 also had inaccessible PCIe
configuration (`rev ff`, `Unknown header type 7f`); the initiating fault was
not determined because reading kernel logs requires administrator access.

```bash
screen -r qt1
# Detach without stopping: Ctrl-a, then d
cat /home/user/toy_hybrid_20260912/scheduler_status.json
# After an administrator has rebooted the host, start the screen session again:
bash /home/user/toy_hybrid_20260912/start_qt1.sh
```

`results.md` refreshes every 15 seconds. Each run writes `train.log`,
`history.jsonl`, `status.json`, and an atomic `latest.pt` checkpoint every
25 epochs. Final results go to `result.json`, `predictions.npz`, and
`masked_predictions.npz`. Restarting the supervisor resumes saved runs.
Stop only this supervisor with Ctrl-C in its screen window if needed;
already-running worker processes require explicit handling.

The code/data validation checks common initialization, absence of future-token
leakage (including convolution), finite gradients, equivalence to recurrent
linear attention, training/test disjointness, forbidden pairs, loss weights,
and semantic evaluation targets. GPU throughput and end-to-end training are
not verified until CUDA becomes available.
