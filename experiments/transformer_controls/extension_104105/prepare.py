"""Validate new datasets without changing the historical implementation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).parent.resolve()
SOURCE = ROOT.parent
sys.path.insert(0, str(SOURCE))
from data import make_dataset, EXCLUDED, SEP
from train import refresh_noise


def main():
    torch.set_num_threads(2)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seeds': {}}
    for seed in [104, 105]:
        path = make_dataset(seed, SOURCE/'datasets')
        meta = json.loads(path.with_suffix('.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256']
        assert meta['stable_arithmetic_examples'] == 270000
        assert meta['retrieval_training_examples'] == 135000
        assert meta['random_operator_examples'] == 180000
        with np.load(path) as z:
            x, y, w = (z['base_'+k] for k in ['x', 'y', 'w'])
            assert len(x) == 405000
            assert np.array_equal(x[:, 1:], y[:, :-1])
            for a, b in EXCLUDED:
                ret = x[270000:]
                assert not np.any((ret[:, :-1] == a) & (ret[:, 1:] == b))
            assert np.all(z['noise_x'][:, 2] == 9) and np.all(z['noise_x'][:, 3] == 10)
            assert np.all(z['noise_w'][:, 2] == 1)
            assert int(((z['noise_x'][:, :-1] == 9) & (z['noise_x'][:, 1:] == 10)).sum()) == 180000
            for train, test in zip(z['train_operand_ids'], z['held_operand_ids']):
                assert not np.intersect1d(train, test).size
            assert not np.intersect1d(z['noise_train_operand_ids'], z['noise_held_operand_ids']).size
            # Use all noise examples to validate the actual label-refresh routine.
            arrays = {k: torch.from_numpy(np.concatenate([z['base_'+k], z['noise_'+k]])) for k in ['x', 'y', 'w']}
            first = refresh_noise(arrays, 405000, seed, 0).clone()
            assert torch.equal(first, refresh_noise(arrays, 405000, seed, 0))
            second = refresh_noise(arrays, 405000, seed, 1).clone()
            changed = float((first != second).float().mean())
            assert changed > .97
            assert torch.equal(arrays['x'][405000:, 5], arrays['y'][405000:, 4])
            assert torch.equal(arrays['x'][:, 1:], arrays['y'][:, :-1])
            assert torch.equal(arrays['x'][:405000], torch.from_numpy(x))
            for name in [k[:-7] for k in z.files if k.endswith('_answer')]:
                gold = z[name+'_answer']
                assert np.array_equal(z[name+'_y'][np.arange(len(gold)), z[name+'_pos']], gold)
            assert len(z['ordinary_9_10_answer']) == 2560
            assert np.all(z['ordinary_9_10_answer'] == 10)
            assert np.all(z['ordinary_9_10_first'] == 9)
            report['seeds'][str(seed)] = {
                'dataset_sha256': meta['sha256'], 'base_examples': 405000, 'noise_examples': 180000,
                'arithmetic_examples': 270000, 'retrieval_examples': 135000,
                'retrieval_training_target_pair_occurrences': 0,
                'noise_training_target_pair_occurrences': 180000,
                'random_label_refresh_change_fraction': changed,
                'random_labels_deterministic_per_seed_epoch': True,
                'input_target_shift_consistent': True, 'test_target_positions_correct': True}
        print('PASS dataset', seed, flush=True)
    (ROOT/'validation.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
