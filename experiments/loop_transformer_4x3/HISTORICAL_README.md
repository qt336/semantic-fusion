# Loop Transformer: four layers, three loops

Follow-up experiment requested on 2026-10-03. Four Transformer blocks are reused
three times, giving 12 effective layers and 3,331,920 unique parameters. Every
visit remains in the gradient graph. Only the number of loops changes from the
paired loop-2 experiment; training starts from fresh initialization for each seed.

Seeds 101–105 reuse the original checksum-verified datasets. Each seed runs for
1,200 epochs with effective batch 8,190 and microbatch 2,048. AdamW, warmup,
cosine learning rate 1e-3 to 1e-4, loss weights, FP32/TF32, initialization,
evaluation frequency and checkpoint retention match the loop-2 protocol.
See `protocol.json` for all settings. No gradient clipping or adaptive halting.

## Queue and execution

`chain_after_loop2.py` waits until **all five** predecessor seeds have complete
epoch-1200 results, the predecessor scheduler reports complete, and its scheduler
lock is released. It then launches `launch.py --gpus 2 3` from the same `qt2`
screen session's `loop3_pending` window. The scheduler also waits for assigned
GPUs to become idle and pass a CUDA probe before starting each worker.
Failed or stopped predecessor experiments do not satisfy the start condition.

Predecessor: `/home/user/toy_loop_4x2_20261003`.

Queue check (does not start training):

```bash
/usr/bin/python3 /home/user/toy_loop_4x3_20261003/chain_after_loop2.py --check-only
```

`scheduler_status.json` records the dependency and pending seeds while waiting.
After launch, it records running seeds and GPU assignments. `chain.log` records
dependency changes; `supervisor.log` records scheduler output. Training outputs
are under `runs/loop_seed<seed>/`, separate from the loop-2 results.

`validate.py` checks three visits to each of the four blocks, equivalence to an
unrolled 12-layer Transformer with copied weights, summed gradients across all
three visits, causality, initialization and paired dataset hashes.
