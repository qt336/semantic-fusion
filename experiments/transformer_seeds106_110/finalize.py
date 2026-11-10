"""Independently recount all final baseline and masked saved predictions."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).parent.resolve()


def main():
    rows = []
    for seed in range(106, 111):
        directory = ROOT / 'runs' / f'transformer_seed{seed}'
        result = json.loads((directory / 'result.json').read_text())
        assert result['completed_epochs'] == 1200 and not result['small_smoke_test']
        assert result['architecture'] == 'transformer'
        with np.load(ROOT / 'datasets' / f'seed_{seed}.npz') as dataset:
            for filename, key in [('predictions.npz', 'baseline'), ('masked_predictions.npz', 'layer0_attention_edge_masked')]:
                with np.load(directory / filename) as saved:
                    for name in saved.files:
                        pred, gold, first = saved[name], dataset[name + '_answer'], dataset[name + '_first']
                        correct = pred == gold
                        wrong_first = (pred == first) & ~correct
                        eos = (pred == 334) & ~correct & ~wrong_first
                        other = ~(correct | wrong_first | eos)
                        counts = {k: int(v.sum()) for k, v in [('correct', correct), ('first', wrong_first), ('eos', eos), ('other', other)]}
                        assert counts == result[key][name]['counts']
                        assert result[key][name]['n'] == len(pred)
                        rows.append({'seed': seed, 'condition': name, 'evaluation': key, 'counts': counts, 'n': len(pred)})
    (ROOT / 'final_prediction_verification.json').write_text(json.dumps({'passed': True, 'rows': rows}, indent=2) + '\n')
    print(f'Final saved predictions verified: {len(rows)} conditions.', flush=True)


if __name__ == '__main__':
    main()
