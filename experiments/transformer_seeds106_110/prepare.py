"""Generate and audit the unchanged initial mixed task for seeds106--110."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from data import make_dataset, OPS, BOS, EOS, SEP, QUERY, PAD, valid_memory
from model import ToyModel

ROOT = Path(__file__).parent.resolve()
REFERENCE = ROOT.parent / 'toy_transformer_controls_20261018/reference_20260912'
ORIGINAL = ROOT.parent / 'qintian/toy_hybrid_20260912'
SEEDS = list(range(106, 111))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signatures(x, y, pos, seen=None):
    seen = set() if seen is None else seen
    for row, targets, p in zip(x, y, pos):
        n = int(p) - 3
        memory = row[1:n + 1].tolist()
        idx = memory.index(int(row[p]))
        assert 3 <= n <= 6 and len(set(memory)) == n and idx > 0
        assert row[0] == BOS and row[n + 1] == SEP and row[n + 2] == QUERY
        assert targets[p] == memory[idx - 1] and targets[p + 1] == EOS
        assert valid_memory(memory, idx)
        sig = (*memory, -1, idx)
        assert sig not in seen
        seen.add(sig)
    return seen


def audit(path):
    with np.load(path) as d:
        x, y, w = (d['train_' + k] for k in ['x', 'y', 'w'])
        assert x.shape == y.shape == w.shape == (405000, 11)
        for j, (o1, o2) in enumerate(OPS):
            xx = x[j * 90000:(j + 1) * 90000]
            assert np.all(xx[:, 2:4] == [o1, o2])
            a, b = xx[:, 1], xx[:, 4]
            result = (a + b if j == 0 else a - b if j == 1 else a - b - 1) % 331
            assert np.array_equal(xx[:, 5], result) and np.all(xx[:, 6] == EOS)
            assert np.array_equal(a * 331 + b, d['train_operand_ids'][j])
            assert len(np.unique(d['train_operand_ids'][j])) == 90000
            assert len(np.unique(d['held_operand_ids'][j])) == 4096
            assert len(np.intersect1d(d['train_operand_ids'][j], d['held_operand_ids'][j])) == 0
            assert np.allclose(w[j * 90000:(j + 1) * 90000], [0.1, 1, 1, 0.1, 10, 1, 0, 0, 0, 0, 0])
            prefix = f'semantic_{j}'
            p = d[prefix + '_pos']
            assert len(p) == 2560 and all(int((p == n + 3).sum()) == 640 for n in range(3, 7))
            edge = d[prefix + '_edge']
            rows = np.arange(2560)
            assert np.all(d[prefix + '_x'][rows, edge[:, 1]] == o1)
            assert np.all(d[prefix + '_x'][rows, edge[:, 0]] == o2)
            assert np.all(d[prefix + '_answer'] == o2)
            assert np.all(d[prefix + '_y'][rows, p] == o2)
        ret_x, ret_y, ret_w = x[270000:], y[270000:], w[270000:]
        ret_pos = np.argmax(ret_x == QUERY, axis=1) + 1
        seen = signatures(ret_x, ret_y, ret_pos)
        signatures(d['standard_x'], d['standard_y'], d['standard_pos'], seen)
        for n in range(3, 7):
            rows = ret_w[ret_pos == n + 3]
            expected = [*([0.1] * n), 1, 1, 2, 10, 1] + [0] * (6 - n)
            assert np.allclose(rows, expected)
        assert np.all(w[y == PAD] == 0)
        counts = {key[:-7]: len(d[key]) for key in d.files if key.endswith('_answer')}
        assert counts == {'arithmetic_0': 4096, 'arithmetic_1': 4096, 'arithmetic_2': 4096,
                          'standard': 2000, 'semantic_0': 2560, 'semantic_1': 2560,
                          'semantic_2': 2560, 'control_a1b': 2560, 'control_a2b': 2560}
    return {'passed': True, 'training_examples': 405000, 'arithmetic_examples': 270000,
            'retrieval_examples': 135000, 'evaluation_counts': counts, 'sha256': digest(path)}


def main():
    for name in ['data.py', 'model.py', 'train.py']:
        assert digest(ROOT / name) == digest(REFERENCE / name)
    # Compare all generated arrays, not NPZ container bytes, to the saved September data.
    path = make_dataset(101, ROOT / 'reference_check', small=True)
    with np.load(path) as current, np.load(ORIGINAL / 'datasets/seed_101_small.npz') as old:
        assert set(current.files) == set(old.files)
        for key in current.files:
            assert np.array_equal(current[key], old[key]), key
        reference_arrays = len(current.files)
    model = ToyModel('transformer')
    assert len(model.blocks) == 4 and all(b.mixer.conv is None and b.mixer.kind == 'softmax' for b in model.blocks)
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 3331920
    checks = {}
    for seed in SEEDS:
        path = make_dataset(seed, ROOT / 'datasets')
        checks[str(seed)] = audit(path)
        print(f'Seed {seed}: data generation and audit passed', flush=True)
    manifest = {'task': 'initial arithmetic + retrieval mixed task, plain Transformer',
                'seeds': SEEDS, 'epochs': 1200, 'batch_size': 8190, 'microbatch': 2048,
                'eval_every': 25, 'checkpoint_every': 25, 'allowed_physical_gpus': [4, 5],
                'parameters': parameters, 'steps_per_epoch': 50, 'total_optimizer_steps': 60000,
                'training_examples_per_epoch': 405000, 'retrieval_excluded_pairs': OPS,
                'additional_random_output_or_cooccurrence_examples': 0,
                'reference_source': str(REFERENCE), 'reference_source_sha256':
                    {n: digest(ROOT / n) for n in ['data.py', 'model.py', 'train.py']},
                'september_seed101_small_all_arrays_identical': reference_arrays,
                'dataset_checks': checks, 'torch_version': torch.__version__,
                'implementation_note': 'Reconstructed implementation from September; original author code was unavailable.',
                'evaluation_note': 'Report fixed epoch1200; keep every recorded epoch and evaluation, no selection by test accuracy.'}
    (ROOT / 'protocol_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('All five full datasets validated.', flush=True)


if __name__ == '__main__':
    main()
