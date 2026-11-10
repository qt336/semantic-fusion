"""Reconstruction from the paper; original code/data were unavailable."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

P, QUERY, SEP, BOS, EOS, PAD = 331, 331, 332, 333, 334, 335
OPS = [(0, 1), (0, 2), (2, 1)]
FORBIDDEN = set(OPS)


def pack(rows, weights):
    seq = np.full((len(rows), 12), PAD, dtype=np.int64)
    w = np.zeros((len(rows), 12), dtype=np.float32)
    for i, (r, rw) in enumerate(zip(rows, weights)):
        seq[i, :len(r)], w[i, :len(r)] = r, rw
    return seq[:, :-1].copy(), seq[:, 1:].copy(), w[:, 1:].copy()


def ret_row(memory, target_idx):
    target, answer = memory[target_idx], memory[target_idx - 1]
    row = [BOS, *memory, SEP, QUERY, target, answer, EOS]
    weight = [0, *([0.1] * len(memory)), 1, 1, 2, 10, 1]
    return row, weight


def valid_memory(mem, idx):
    return (not any(pair in FORBIDDEN for pair in zip(mem, mem[1:]))
            and (mem[idx], mem[idx - 1]) not in FORBIDDEN)


def retrieval(rng, count, used):
    rows, weights, positions, answers = [], [], [], []
    while len(rows) < count:
        n = int(rng.integers(3, 7))
        mem = rng.choice(P, n, replace=False).tolist()
        idx = int(rng.integers(1, n))
        sig = (*mem, -1, idx)
        if sig in used or not valid_memory(mem, idx):
            continue
        used.add(sig)
        row, weight = ret_row(mem, idx)
        rows.append(row); weights.append(weight)
        positions.append(n + 3); answers.append(mem[idx - 1])
    return rows, weights, positions, answers


def make_dataset(seed, root, small=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'seed_{seed}{"_small" if small else ""}.npz'
    if path.exists():
        return path
    rng = np.random.default_rng(seed)
    train_n, held_n, ret_n = (1500, 128, 1500) if small else (90000, 4096, 135000)
    standard_n, pair_n, ctrl_n = (100, 64, 64) if small else (2000, 2560, 2560)
    rows, weights = [], []
    eval_sets = {}
    train_operand_ids, held_operand_ids = [], []
    for op_idx, (o1, o2) in enumerate(OPS):
        perm = rng.permutation(P * P)
        held_ids, train_ids = perm[:held_n], perm[held_n:held_n + train_n]
        train_operand_ids.append(train_ids); held_operand_ids.append(held_ids)
        for name, ids in [('train', train_ids), (f'arithmetic_{op_idx}', held_ids)]:
            a, b = ids // P, ids % P
            c = ((a + b) if op_idx == 0 else (a - b if op_idx == 1 else a - b - 1)) % P
            if not small and name == 'train':
                assert all(len(np.unique(v)) == P for v in [a, b, c])
            rr = [[BOS, int(av), o1, o2, int(bv), int(cv), EOS] for av, bv, cv in zip(a, b, c)]
            ww = [[0, 0.1, 1, 1, 0.1, 10, 1] for _ in ids]
            if name == 'train':
                rows.extend(rr); weights.extend(ww)
            else:
                eval_sets[name] = (rr, ww, [4] * len(rr), c, [o1] * len(rr), [[3, 2]] * len(rr))
    used = set()
    rr, ww, _, _ = retrieval(rng, ret_n, used)
    rows.extend(rr); weights.extend(ww)
    rr, ww, pos, ans = retrieval(rng, standard_n, used)
    eval_sets['standard'] = (rr, ww, pos, ans, [-1] * len(rr), [[1, 0]] * len(rr))
    pool = np.arange(3, P)
    for op_idx, (o1, o2) in enumerate(OPS):
        rr, ww, pos, ans, first, edges = [], [], [], [], [], []
        for i in range(pair_n):
            n = 3 + i % 4
            start = int(rng.integers(n - 2))
            other = rng.choice(pool, n - 2, replace=False).tolist()
            target = other[start]
            mem = other[:start] + [o1, o2] + other[start:]
            row, weight = ret_row(mem, start + 2)
            assert mem[start + 2] == target
            rr.append(row); ww.append(weight); pos.append(n + 3); ans.append(o2)
            first.append(o1); edges.append([start + 2, start + 1])
        eval_sets[f'semantic_{op_idx}'] = (rr, ww, pos, ans, first, edges)
    for answer in [1, 2]:
        rr, ww, pos, ans, first, edges = [], [], [], [], [], []
        seen = set()
        while len(rr) < ctrl_n:
            a, q = rng.choice(pool, 2, replace=False).tolist()
            if (a, q) in seen:
                continue
            seen.add((a, q))
            row, weight = ret_row([a, answer, q], 2)
            rr.append(row); ww.append(weight); pos.append(6); ans.append(answer)
            first.append(a); edges.append([2, 1])
        eval_sets[f'control_a{answer}b'] = (rr, ww, pos, ans, first, edges)
    out = {}
    for suffix, arr in zip(['x', 'y', 'w'], pack(rows, weights)):
        out[f'train_{suffix}'] = arr
    for name, (rr, ww, pos, ans, first, edges) in eval_sets.items():
        for suffix, arr in zip(['x', 'y', 'w'], pack(rr, ww)):
            out[f'{name}_{suffix}'] = arr
        for suffix, arr in [('pos', pos), ('answer', ans), ('first', first), ('edge', edges)]:
            out[f'{name}_{suffix}'] = np.asarray(arr, dtype=np.int64)
    out['train_operand_ids'] = np.stack(train_operand_ids)
    out['held_operand_ids'] = np.stack(held_operand_ids)
    np.savez_compressed(path, **out)
    metadata = {'seed': seed, 'small': small, 'training_examples': len(rows),
                'evaluation_counts': {k: len(v[0]) for k, v in eval_sets.items()},
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'reconstructed_from_paper': True, 'original_data_available': False}
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='outputs/toy/datasets')
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(101, 106)))
    p.add_argument('--small', action='store_true')
    args = p.parse_args()
    for seed in args.seeds:
        print(make_dataset(seed, args.root, args.small), flush=True)
