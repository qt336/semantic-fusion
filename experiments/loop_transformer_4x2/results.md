# Loop Transformer: 4 layers x 2 loops

Updated: 2026-10-04T12:58:33.694077+08:00
State: complete
GPU policy: wait_until_idle; GPUs 2, 3; screen qt2.

| Seed | State | GPU | Epoch | Loss | Arithmetic | Standard | Semantic | First-token error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | complete | — | 1200 | — | 99.87% | 99.90% | 27.88% | 52.24% |
| 102 | complete | — | 1200 | — | 99.98% | 95.95% | 25.55% | 63.74% |
| 103 | complete | — | 1200 | — | 100.00% | 99.90% | 21.81% | 38.10% |
| 104 | complete | — | 1200 | — | 100.00% | 99.90% | 35.72% | 56.21% |
| 105 | complete | — | 1200 | — | 99.92% | 99.75% | 3.88% | 23.96% |

## Final epoch-1200 results (5/5 seeds)

| Metric | Mean | Sample SD |
|---|---:|---:|
| arithmetic | 99.95% | 0.06% |
| standard | 99.08% | 1.75% |
| semantic | 22.97% | 11.82% |
| semantic_0 | 7.72% | 4.94% |
| semantic_1 | 51.80% | 31.41% |
| semantic_2 | 9.38% | 8.03% |
| control_a1b | 99.02% | 2.03% |
| control_a2b | 99.65% | 0.45% |
| semantic_first_error | 46.85% | 15.83% |
