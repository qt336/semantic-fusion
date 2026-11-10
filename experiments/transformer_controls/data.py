"""Paired datasets: retrieval-held-out ordinary pairs and a noisy operator.

The two conditions share exactly the same base training and evaluation arrays.
Random-operator labels stored here are placeholders and MUST be refreshed by
the training worker on every epoch, including the shifted input position.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import numpy as np

P, QUERY, SEP, BOS, EOS, PAD = 331, 331, 332, 333, 334, 335
OPS = [(0, 1), (0, 2), (2, 1)]
ORDINARY = [(9, 10), (15, 20)]
EXCLUDED = set(OPS + ORDINARY)
CONDITIONS = ['heldout_pairs', 'random_operator']
VERSION = 1


def pack(rows, weights):
    seq = np.full((len(rows), 12), PAD, dtype=np.int64)
    w = np.zeros((len(rows), 12), dtype=np.float32)
    for i, (r, rw) in enumerate(zip(rows, weights)):
        seq[i, :len(r)], w[i, :len(r)] = r, rw
    return seq[:, :-1].copy(), seq[:, 1:].copy(), w[:, 1:].copy()


def ret_row(memory, idx):
    return ([BOS, *memory, SEP, QUERY, memory[idx], memory[idx - 1], EOS],
            [0, *([0.1] * len(memory)), 1, 1, 2, 10, 1])


def valid_memory(memory, idx):
    return (not any(pair in EXCLUDED for pair in zip(memory, memory[1:]))
            and (memory[idx], memory[idx - 1]) not in EXCLUDED)


def retrieval(rng, count, used):
    rows, weights, pos, ans = [], [], [], []
    while len(rows) < count:
        n = int(rng.integers(3, 7))
        mem = rng.choice(P, n, replace=False).tolist()
        idx = int(rng.integers(1, n))
        sig = (*mem, -1, idx)
        if sig in used or not valid_memory(mem, idx):
            continue
        used.add(sig)
        r, w = ret_row(mem, idx)
        rows.append(r); weights.append(w); pos.append(n + 3); ans.append(mem[idx - 1])
    return rows, weights, pos, ans


def audit_retrieval(rows):
    adjacent = {f'{a}_{b}': 0 for a, b in OPS + ORDINARY}
    reverse = dict(adjacent)
    memory_counts = np.zeros(P, dtype=np.int64)
    query_counts, answer_counts = memory_counts.copy(), memory_counts.copy()
    signatures = set()
    for r in rows:
        sep = r.index(SEP)
        mem, query, ans = r[1:sep], r[sep + 2], r[sep + 3]
        idx = mem.index(query)
        assert idx > 0 and ans == mem[idx - 1]
        sig = (*mem, -1, idx)
        assert sig not in signatures
        signatures.add(sig)
        for pair in zip(mem, mem[1:]):
            if pair in EXCLUDED: adjacent[f'{pair[0]}_{pair[1]}'] += 1
        if (query, ans) in EXCLUDED: reverse[f'{query}_{ans}'] += 1
        np.add.at(memory_counts, mem, 1)
        query_counts[query] += 1; answer_counts[ans] += 1
    assert sum(adjacent.values()) == sum(reverse.values()) == 0
    for token in {x for pair in OPS + ORDINARY for x in pair}:
        assert memory_counts[token] and query_counts[token] and answer_counts[token]
    return {'unique_examples': len(signatures), 'excluded_memory_pair_counts': adjacent,
            'excluded_query_answer_counts': reverse,
            'constituent_counts': {str(t): {'memory': int(memory_counts[t]),
                  'query': int(query_counts[t]), 'answer': int(answer_counts[t])}
                for t in sorted({x for pair in OPS + ORDINARY for x in pair})}}


def make_dataset(seed, root, small=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    path = root / f'seed{seed}{"_small" if small else ""}.npz'
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        meta_path = path.with_suffix('.json')
        if path.exists() and meta_path.exists():
            assert json.loads(meta_path.read_text())['version'] == VERSION
            return path
        return generate(seed, path, small)


def generate(seed, path, small):
    rng = np.random.default_rng(seed)
    per_op, held_n, ret_n = (500, 128, 3000) if small else (90000, 4096, 135000)
    standard_n, ctrl_n = (100, 64) if small else (2000, 2560)
    # Unique, lexically/positionally paired evaluation templates for all five pairs.
    length_counts = {3: 16, 4: 16, 5: 16, 6: 16} if small else {3: 256, 4: 768, 5: 768, 6: 768}
    rows, weights, evaluations = [], [], {}
    train_ids, held_ids = [], []
    for op_idx, (a, b) in enumerate(OPS):
        perm = rng.permutation(P * P)
        test, train = perm[:held_n], perm[held_n:held_n + per_op]
        train_ids.append(train); held_ids.append(test)
        assert not np.intersect1d(train, test).size
        for is_train, ids in [(True, train), (False, test)]:
            r, s = ids // P, ids % P
            result = ((r + s) if op_idx == 0 else (r - s if op_idx == 1 else r - s - 1)) % P
            rr = [[BOS, int(x), a, b, int(y), int(z), EOS] for x, y, z in zip(r, s, result)]
            ww = [[0, 0.1, 1, 1, 0.1, 10, 1] for _ in rr]
            if is_train:
                rows.extend(rr); weights.extend(ww)
            else:
                evaluations[f'arithmetic_{a}_{b}'] = (rr, ww, [4]*len(rr), result, [-1]*len(rr), [-1]*len(rr))
    arithmetic_rows = len(rows)
    used = set()
    rr, ww, _, _ = retrieval(rng, ret_n, used)
    audit = audit_retrieval(rr)
    rows.extend(rr); weights.extend(ww)
    rr, ww, pos, ans = retrieval(rng, standard_n, used)
    evaluations['standard'] = (rr, ww, pos, ans, [-1]*len(rr), [r[r.index(SEP)+2] for r in rr])
    pool = np.array([x for x in range(3, P) if x not in {9, 10, 15, 20}])
    templates, seen = [], set()
    for n, count in length_counts.items():
        accepted = 0
        while accepted < count:
            start = int(rng.integers(n - 2))
            other = rng.choice(pool, n - 2, replace=False).tolist()
            signature = (n, start, *other)
            if signature in seen: continue
            seen.add(signature); templates.append((n, start, other)); accepted += 1
    for a, b in OPS + ORDINARY:
        rr, ww, pos, ans, first, query = [], [], [], [], [], []
        for n, start, other in templates:
            mem = other[:start] + [a, b] + other[start:]
            r, w = ret_row(mem, start + 2)
            rr.append(r); ww.append(w); pos.append(n+3); ans.append(b)
            first.append(a); query.append(other[start])
        label = 'operator' if (a, b) in OPS else 'ordinary'
        evaluations[f'{label}_{a}_{b}'] = (rr, ww, pos, ans, first, query)
    for answer in [1, 2]:
        rr, ww, pos, ans, first, query = [], [], [], [], [], []
        seen = set()
        while len(rr) < ctrl_n:
            x, q = map(int, rng.choice(pool, 2, replace=False))
            if (x, q) in seen: continue
            seen.add((x, q)); r, w = ret_row([x, answer, q], 2)
            rr.append(r); ww.append(w); pos.append(6); ans.append(answer); first.append(x); query.append(q)
        evaluations[f'control_x{answer}y'] = (rr, ww, pos, ans, first, query)
    out = {}
    for suffix, arr in zip(['x', 'y', 'w'], pack(rows, weights)): out['base_'+suffix] = arr
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 910, 20261018]))
    operand_perm = noise_rng.permutation(P * P)
    noise_held_ids = operand_perm[:held_n]
    noise_train_ids = noise_rng.choice(operand_perm[held_n:], 2 * per_op, replace=True)
    assert not np.intersect1d(noise_train_ids, noise_held_ids).size
    for is_train, ids in [(True, noise_train_ids), (False, noise_held_ids)]:
        target = np.zeros(len(ids), dtype=int) if is_train else noise_rng.integers(P, size=len(ids))
        rr = [[BOS, int(i//P), 9, 10, int(i%P), int(t), EOS] for i, t in zip(ids, target)]
        ww = [[0, 0.1, 1, 1, 0.1, 10, 1] for _ in rr]
        if is_train:
            for suffix, arr in zip(['x', 'y', 'w'], pack(rr, ww)): out['noise_'+suffix] = arr
        else:
            evaluations['noise_arithmetic_9_10'] = (rr, ww, [4]*len(rr), target, [-1]*len(rr), [-1]*len(rr))
    for name, (rr, ww, pos, ans, first, query) in evaluations.items():
        for suffix, arr in zip(['x', 'y', 'w'], pack(rr, ww)): out[name+'_'+suffix] = arr
        for suffix, arr in [('pos', pos), ('answer', ans), ('first', first), ('query', query)]:
            out[name+'_'+suffix] = np.asarray(arr, dtype=np.int64)
        assert len({tuple(r) for r in rr}) == len(rr), name
    out['train_operand_ids'] = np.stack(train_ids)
    out['held_operand_ids'] = np.stack(held_ids)
    out['noise_train_operand_ids'], out['noise_held_operand_ids'] = noise_train_ids, noise_held_ids
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as f: np.savez_compressed(f, **out)
    temporary.replace(path)
    metadata = {'version': VERSION, 'seed': seed, 'small': small,
        'base_training_examples': len(rows), 'stable_arithmetic_examples': arithmetic_rows,
        'retrieval_training_examples': ret_n, 'random_operator_examples': len(noise_train_ids),
        'operator_slot_exposures_per_epoch': {**{f'{a}_{b}': per_op for a,b in OPS}, '9_10_random_condition': 2*per_op},
        'retrieval_audit': audit, 'evaluation_counts': {k: len(v[0]) for k,v in evaluations.items()},
        'pair_evaluation_memory_length_counts': length_counts, 'pair_evaluation_unique': True,
        'shared_contexts_across_pair_evaluations': True,
        'noise_result_policy': 'uniform 0..330, redraw at each epoch/occurrence; stored training zeros are placeholders',
        'noise_result_y_position': 4, 'noise_result_x_position': 5,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'reconstructed_from_paper': True}
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=str(Path(__file__).parent/'datasets'))
    p.add_argument('--seeds', nargs='+', type=int, default=[101,102,103])
    p.add_argument('--small', action='store_true')
    args = p.parse_args()
    for seed in args.seeds: print(make_dataset(seed,args.root,args.small),flush=True)
