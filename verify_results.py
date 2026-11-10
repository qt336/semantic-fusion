"""Recount saved predictions and LLM responses without training or API calls."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'supplementary'))
from llm.common import load_items
from llm.scoring import proposed_answer, outcome


def read_json(path):
    return json.loads(path.read_text())


def verify_llm():
    items = load_items(ROOT / 'supplementary/llm/data/items.jsonl')
    summaries = read_json(ROOT / 'results/llm/summary.json')
    total = 0
    for model in ['kimi', 'deepseek', 'hy4']:
        rows = [json.loads(line) for line in (ROOT / f'results/llm/{model}.jsonl').read_text().splitlines()]
        assert len(rows) == len({r['sample_id'] for r in rows}) == 3000
        counts = Counter()
        for row in rows:
            item = items[row['sample_id']]
            assert all(row[k] == item[k] for k in ['language', 'group', 'a', 'b', 'c'])
            answer, _ = proposed_answer(dict(item, answer_raw=row['answer_raw']))
            label = outcome(item, answer)
            assert label == outcome(item, row['first_answer']), (model, row['sample_id'])
            assert row['correct'] == (label == 'B')
            counts[(row['language'], row['group'], label)] += 1
            total += 1
        for summary in summaries:
            if summary['model'] == model:
                assert summary['n'] == 500
                for label in ['B', 'A', 'Q', 'Other']:
                    assert counts[(summary['language'], summary['group'], label)] == summary[label]
    # Check the percentages against the exact paper-table source.
    for lang, number in [('en', 6), ('zh', 7)]:
        path = next((ROOT / 'results/paper_tables').glob(f'table_{number:02d}_*.tex'))
        actual = []
        for line in path.read_text().splitlines():
            if re.search(r'&\s*(Semantic|Random|Comma)\s*&', line):
                actual.append([float(re.sub(r'\\textbf\{([^}]+)\}', r'\1', cell).strip().rstrip('\\').strip()) for cell in line.split('&')[2:]])
        expected = [[100 * s[k] / s['n'] for k in ['B', 'A', 'Q', 'Other']] for model in ['kimi', 'deepseek', 'hy4'] for group in ['semantic', 'random', 'comma_semantic'] for s in summaries if (s['model'], s['language'], s['group']) == (model, lang, group)]
        assert np.allclose(actual, expected), f'Paper table {number} differs'
    return total


def verify_toy():
    runs = []
    conditions = 0
    predictions = 0
    for folder in ['loop_transformer_4x2', 'loop_transformer_4x3', 'transformer_seeds106_110', 'transformer_controls', 'hybrid_seed104_rerun']:
        experiment = ROOT / 'experiments' / folder
        for path in sorted((experiment / 'runs').glob('*/result.json')):
            result = read_json(path)
            assert result['completed_epochs'] == 1200 and not result['small_smoke_test']
            seed = result['seed']
            controls = 'metrics' in result
            data_root = ROOT / 'experiments/toy_hybrid/datasets' if folder.startswith('loop_') else experiment / 'datasets'
            data_path = data_root / (f'seed{seed}.npz' if controls else f'seed_{seed}.npz')
            metadata = read_json(data_path.with_suffix('.json'))
            assert hashlib.sha256(data_path.read_bytes()).hexdigest() == metadata['sha256']
            baseline = result['metrics'] if controls else result['baseline']
            variants = [('predictions.npz', baseline)]
            if result.get('layer0_attention_edge_masked'):
                variants.append(('masked_predictions.npz', result['layer0_attention_edge_masked']))
            with np.load(data_path, allow_pickle=False) as dataset:
                for filename, metrics in variants:
                    with np.load(path.parent / filename, allow_pickle=False) as saved:
                        assert set(saved.files) == {name for name in metrics if name + '_answer' in dataset.files}
                        for name in saved.files:
                            pred = saved[name]
                            correct = pred == dataset[name + '_answer']
                            first = (pred == dataset[name + '_first']) & ~correct
                            masks = {'correct': correct, 'first': first}
                            used = correct | first
                            if controls:
                                masks['query'] = (pred == dataset[name + '_query']) & ~used
                                used |= masks['query']
                            masks['eos'] = (pred == 334) & ~used
                            masks['other'] = ~(used | masks['eos'])
                            counts = {k: int(v.sum()) for k, v in masks.items()}
                            metric = metrics[name]
                            assert counts == metric['counts'], (folder, seed, name, filename)
                            assert len(pred) == metric['n']
                            assert math.isclose(metric['accuracy'], counts['correct'] / len(pred))
                            assert all(math.isclose(metric['rates'][k], v / len(pred)) for k, v in counts.items())
                            conditions += 1
                            predictions += len(pred)
                        for name, metric in metrics.items():
                            if name in saved.files:
                                continue
                            components = [metrics[k] for k in saved.files if k.startswith(name + '_')]
                            assert components, (folder, seed, name)
                            assert sum(x['n'] for x in components) == metric['n']
                            assert all(sum(x['counts'][k] for x in components) == v for k, v in metric['counts'].items())
                            assert math.isclose(metric['accuracy'], metric['counts']['correct'] / metric['n'])
            semantic = baseline['operator' if controls else 'semantic']
            runs.append({'experiment': folder, 'run': path.parent.name, 'seed': seed, 'epoch': 1200,
                         'arithmetic_pct': 100 * baseline['arithmetic']['accuracy'],
                         'standard_pct': 100 * baseline['standard']['accuracy'],
                         'semantic_pct': 100 * semantic['accuracy'],
                         'semantic_first_error_pct': 100 * semantic['rates']['first']})
    return runs, conditions, predictions


def verify_manifest():
    path = ROOT / 'MANIFEST.sha256'
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report', action='store_true', help='Refresh publication report and toy-summary CSV.')
    args = parser.parse_args()
    llm_rows = verify_llm()
    runs, conditions, predictions = verify_toy()
    tables = read_json(ROOT / 'results/paper_tables/index.json')
    assert sorted(t['number'] for t in tables) == list(range(1, 20))
    for table in tables:
        assert hashlib.sha256((ROOT / table['path']).read_bytes()).hexdigest() == table['sha256'], table['path']
    report = {'status': 'passed', 'llm_responses_rescored': llm_rows, 'toy_runs_verified': len(runs),
              'prediction_conditions_verified': conditions, 'individual_predictions_verified': predictions,
              'paper_tables_preserved': len(tables), 'paper_llm_tables_match_rescoring': True}
    if args.write_report:
        (ROOT / 'results/verification.json').write_text(json.dumps(report, indent=2) + '\n')
        with (ROOT / 'results/toy_runs.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
            writer.writeheader()
            writer.writerows(runs)
    else:
        report['file_checksums_verified'] = verify_manifest()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
