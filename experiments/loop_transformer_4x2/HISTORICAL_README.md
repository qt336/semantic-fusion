# Loop Transformer: four independent layers, two loops

Approved experiment on 2026-10-03. The model applies A → B → C → D → A → B → C → D,
sharing each block's attention, MLP and LayerNorm parameters between its two visits.
Effective depth is eight, with 3,331,920 unique parameters. Gradients propagate
through all visits. Token positions are unchanged across loops; no loop embeddings,
adaptive exits, state detachment, convolution or intermediate-loop loss are added.

## Training protocol

- Width 256; eight heads; MLP width 1,024; GELU; pre-LayerNorm and final LayerNorm.
- Causal softmax attention, interleaved RoPE base 10,000, vocabulary 336.
- PyTorch default initialization; bias enabled; untied embedding/output weights.
- Existing paired datasets in `/home/user/toy_hybrid_20260912/datasets` are reused
  and checksum-verified: 270,000 arithmetic + 135,000 predecessor-retrieval examples.
- Seeds 101–105; 1,200 epochs; 50 updates per epoch; 60,000 updates per seed.
- Effective batch 8,190; maximum microbatch 2,048; weighted gradient accumulation.
- AdamW: learning rate 1e-3, betas (0.9, 0.999), eps 1e-8, weight decay 0.1 on all
  parameters, matching the original parameter grouping.
- Linear warmup over 500 updates, cosine decay to 1e-4; residual branch scale 1.
- No dropout or gradient clipping; FP32 with TF32 permitted.
- Weighted next-token CE only after the final loop. Retrieval memory/query/answer
  weights 0.1/2/10; arithmetic operand/result weights 0.1/10; other valid tokens 1;
  padding 0. Normalize by the sum of weights of the effective batch.
- Full sequence length at most 12; shifted training input length 11.
- Evaluation after epoch 1, then every 25 epochs and epoch 1,200. Report arithmetic,
  standard retrieval, semantic pairs, controls, and first-token/EOS/other errors.
- Save every 25-epoch checkpoint separately, with `latest.pt` linked to the newest.
  Main results use epoch 1,200 for every seed, without test-based checkpoint selection.

## Execution and outputs

After the user-requested migration on 2026-10-03, `screen qt2` supervises one
independent seed per physical GPU 2–3. Seeds 101 and 102 run in parallel, with
103–105 queued for the next released assigned GPU. GPUs 0–1 and 4–7 are excluded.
The earlier short runs on GPUs 4–7 had not reached the first checkpoint; their
logs are preserved under `pre_gpu_migration_20261003/`. Without `--allow-sharing`, workers wait until the assigned
GPU has no compute process and low utilization. With that flag, training coexists
with existing processes, provided at least 8 GiB is free. Existing processes are
never terminated. Each selected GPU must pass a real CUDA matrix multiplication.

```bash
bash /home/user/toy_loop_4x2_20261003/start_qt2.sh --gpus 2 3
# For explicitly selected concurrent GPU sharing:
bash /home/user/toy_loop_4x2_20261003/start_qt2.sh --allow-sharing
screen -r qt2
```

`scheduler_status.json` and `results.md` update every 15 seconds. Each seed has
`runs/loop_seed<seed>/config.json`, `assignment.json`, `train.log`, `history.jsonl`,
`status.json`, `checkpoints/`, and eventual `result.json` and `predictions.npz`.
Restarting the same supervisor resumes completed epoch checkpoints. A worker failure
is recorded and is not silently retried. The supervisor exits after all seeds finish.

Code is copied from `toy_hybrid_20260912`, with an explicit loop architecture,
verified paired-data loading, complete checkpoint retention and a new supervisor.
`validate.py` checks forward and summed-gradient equivalence against eight unrolled
layers, verifies each of the four blocks is called twice, checks causal isolation
and common initialization, and verifies all five saved dataset checksums.
