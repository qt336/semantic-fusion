# Loop Transformer: 4 layers x 3 loops

Updated: 2026-10-05T21:52:09.036863+08:00
State: complete
GPU policy: wait_until_idle; GPUs 2, 3; screen qt2.

| Seed | State | GPU | Epoch | Loss | Arithmetic | Standard | Semantic | First-token error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | complete | — | 1200 | — | 60.45% | 99.25% | 1.85% | 22.92% |
| 102 | complete | — | 1200 | — | 99.99% | 99.80% | 24.52% | 28.06% |
| 103 | complete | — | 1200 | — | 99.69% | 99.30% | 17.99% | 56.15% |
| 104 | complete | — | 1200 | — | 99.98% | 99.95% | 19.36% | 68.39% |
| 105 | complete | — | 1200 | — | 99.21% | 99.40% | 6.52% | 23.62% |

## Final epoch-1200 results (5/5 seeds)

| Metric | Mean | Sample SD |
|---|---:|---:|
| arithmetic | 91.87% | 17.56% |
| standard | 99.54% | 0.32% |
| semantic | 14.05% | 9.47% |
| semantic_0 | 5.88% | 5.33% |
| semantic_1 | 30.44% | 27.59% |
| semantic_2 | 5.83% | 2.76% |
| control_a1b | 99.98% | 0.02% |
| control_a2b | 99.07% | 0.73% |
| semantic_first_error | 39.83% | 21.03% |
