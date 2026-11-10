# Toy hybrid: final 5-seed summary (de-spiked)

Generated from `runs/*/result.json` and `runs/hybrid_seed104/history.jsonl`.

## Substitution rule

`hybrid_seed104` finished inside a loss spike: its per-epoch loss was 0.24 up to epoch 1198,
then 2.55 (ep1199) and 2.46 (ep1200), and `latest.pt` keeps only the newest weights, so the
epoch-1200 snapshot is a transient damaged state (arithmetic 3.60%). Its arithmetic had been
99.93% at epoch 1175, where the loss window epochs 1169-1181 is flat at 0.25. Seed 104 is
therefore reported at epoch 1175. Epoch 1150 is not usable either (spike 2.61 at epoch 1146,
arithmetic 20.72%). The other four seeds finished in a settled state and are reported at epoch 1200.

No weights were saved at epoch 1175, so that row comes from the evaluation record written at
that epoch; `runs/hybrid_seed104/predictions.npz` still belongs to the epoch-1200 state.

## Per seed

| Seed | Reported at | Arithmetic | Standard retrieval | Semantic (01/02/21) | Semantic first-token | Controls a1b/a2b |
|---:|---|---:|---:|---:|---:|---:|
| 101 | result.json @ep1200 | 100.00% | 96.60% | 81.1 / 67.0 / 82.7% | 6.21% | 100.0 / 100.0% |
| 102 | result.json @ep1200 | 100.00% | 96.65% | 83.2 / 72.5 / 74.3% | 15.91% | 99.7 / 100.0% |
| 103 | result.json @ep1200 | 99.99% | 99.50% | 94.1 / 94.0 / 95.6% | 1.73% | 99.6 / 100.0% |
| 104 | history @ep1175 | 99.93% | 97.40% | 50.0 / 79.6 / 76.9% | 19.49% | 99.8 / 97.7% |
| 105 | result.json @ep1200 | 100.00% | 98.35% | 76.9 / 85.2 / 81.6% | 10.01% | 99.2 / 86.8% |

## 5-seed mean

| Metric | Mean | SD | Per seed (101,102,103,104,105) |
|---|---:|---:|---|
| Arithmetic | 99.99% | 0.03 | 100.0 100.0 100.0 99.9 100.0 |
| Standard retrieval | 97.70% | 1.10 | 96.6 96.7 99.5 97.4 98.4 |
| Semantic retrieval | 79.65% | 8.47 | 76.9 76.7 94.6 68.8 81.2 |
| Semantic first-token error | 10.67% | 6.41 | 6.2 15.9 1.7 19.5 10.0 |
| Semantic pair 01 (0,1) | 77.1% (collapse 13.1%) | 14.7 / 13.7 | 81.1 83.2 94.1 50.0 76.9 |
| Semantic pair 02 (0,2) | 79.7% (collapse 8.1%) | 9.4 / 5.8 | 67.0 72.5 94.0 79.6 85.2 |
| Semantic pair 21 (2,1) | 82.2% (collapse 10.8%) | 7.4 / 7.1 | 82.7 74.3 95.6 76.9 81.6 |
| Control a1b | 99.68% | 0.26 | 100.0 99.7 99.6 99.8 99.2 |
| Control a2b | 96.91% | 5.11 | 100.0 100.0 100.0 97.7 86.8 |

## Layer-0 attention-edge ablation (second operator token -> first)

Seed 104 has no clean checkpoint for this ablation (only the spike-contaminated epoch-1200
weights exist), so the masked means below cover seeds 101, 102, 103, 105.

| Seed | Masked arithmetic | Masked semantic |
|---:|---:|---:|
| 101 | 7.23% | 69.84% |
| 102 | 7.12% | 8.52% |
| 103 | 3.91% | 90.86% |
| 104 | n/a (contaminated final state) | n/a |
| 105 | 30.74% | 79.75% |

Masked mean over the four usable seeds: arithmetic 12.25%, semantic 62.24%.

## Reference: transformer baseline, same data, hyperparameters and seeds

`21opB/mixed_retrieval_arithmetic_331` (4L8H d256, 135k retrieval + 270k arithmetic, 1200 epochs,
bf16, 6-GPU DDP), 5 seeds at epoch 1199: arithmetic 99.70%; q01 14.5% (collapse 38.7%),
q02 34.0% (36.4%), q21 11.0% (65.2%).

