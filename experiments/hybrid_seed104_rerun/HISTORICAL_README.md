# Hybrid seed104: fresh rerun with a complete accuracy trace

Requested on 2026-11-07 after the historical raw training log could not be found. This is a NEW training run, not a recovered historical curve. A new seed104 run is not guaranteed to reproduce the old spike or old numerical results.

Uses the unchanged hybrid model and data generator in /home/user/qintian/toy_hybrid_20260912. The exact existing full seed_104.npz is copied, SHA256 b16b83fb0c1692db6ca2eb83010f21999b22433ba73a0c4eca625b82da763a11. The implementation is the prior reconstruction, not original paper Transformer code.

- 4 blocks, width 256, 8 heads, MLP 1024, Pre-LN and RoPE. Blocks 1/3: ELU+1 normalized kernel attention and width-3 depthwise causal convolution. Blocks 2/4: softmax attention. 3,338,064 parameters.
- Seed 104, 1200 epochs, 405000 examples (270000 arithmetic + 135000 retrieval), batch 8190, microbatch 2048.
- AdamW lr 1e-3 cosine to 1e-4, warmup 500 optimizer updates, betas .9/.999, weight decay .1. float32 with TF32 allowed, no dropout or clipping. Original losses and shuffle schedule retained.
- Evaluate every epoch (instead of every 25) so the complete curve and short-lived failures are recorded. No model changes or best-accuracy selection.
- Full optimizer resume checkpoint every 25 epochs (latest.pt); retain weights-only snapshots every 25 epochs and every epoch from 1175 onward. Thus 1175 and 1200 weights will both remain available. Estimated retained snapshots about 1 GB.
- Only the unmasked experiment is needed; final attention-edge ablation is skipped explicitly.
- Run within screen qt1, dedicated window hybrid104_rerun. GPU 1 is shared with current jobs, within the existing GPU 0–3 restriction. No other job is stopped.

results.md and accuracy_curve.png/.pdf refresh from completed epoch records. history.jsonl contains loss and the full evaluation every epoch. Separate curves show arithmetic, standard retrieval, semantic retrieval and each operator pair. No interpolation of missing data. A small smoke run is excluded from the formal curve.

The historical results.md remains unchanged. This rerun's final result is reported at epoch 1200 without substituting an earlier evaluation. If a spike occurs it stays visible in the full trace.
