"""Verify five final saved response sets without rerunning model generation."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).parent.resolve()
SOURCE = ROOT.parent


def main():
    report = {'fixed_final_epoch': 1200, 'seeds': {}}
    for seed in [101,102,103,104,105]:
        run = SOURCE/'runs'/f'random_operator_seed{seed}'
        final = json.loads((run/'result.json').read_text())
        assert final['completed_epochs'] == 1200 and final['small_smoke_test'] is False
        checked = 0
        with np.load(SOURCE/'datasets'/f'seed{seed}.npz') as data, np.load(run/'predictions.npz') as predictions:
            for name in [k[:-7] for k in data.files if k.endswith('_answer')]:
                counts = {k:0 for k in ['correct','first','query','eos','other']}
                for output, gold, a, q in zip(predictions[name],data[name+'_answer'],data[name+'_first'],data[name+'_query']):
                    if output == gold: key = 'correct'
                    elif output == a: key = 'first'
                    elif output == q: key = 'query'
                    elif output == 334: key = 'eos'
                    else: key = 'other'
                    counts[key] += 1
                assert counts == final['metrics'][name]['counts']
                checked += 1
        report['seeds'][str(seed)] = {'all_saved_prediction_counts_match': True, 'metric_groups_checked': checked,
                                    'predictions_sha256': hashlib.sha256((run/'predictions.npz').read_bytes()).hexdigest()}
    (ROOT/'final_verification.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
