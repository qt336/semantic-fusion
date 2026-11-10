"""Score model responses produced by a benchmark run."""
import argparse
import csv
import json
from pathlib import Path

from .common import DATA, load_items, read_jsonl
from .scoring import proposed_answer, outcome


def rescore(items, responses):
    seen, summaries, scored = set(), {}, []
    for response in responses:
        identity = (response['model'], response['sample_id'])
        if identity in seen:
            raise ValueError(f'Duplicate response: {identity}')
        seen.add(identity)
        if response.get('status', 'completed') != 'completed':
            continue
        item = items[response['sample_id']]
        answer, rule = proposed_answer(dict(item, answer_raw=response['answer_raw']))
        label = outcome(item, answer)
        key = ':'.join([item['language'], item['group'], response['model']])
        counts = summaries.setdefault(key, dict(n=0, B=0, A=0, Q=0, Other=0))
        counts['n'] += 1
        counts[label] += 1
        scored.append(dict(sample_id=item['sample_id'], model=response['model'],
                           first_answer=answer, outcome=label, rule=rule))
    return summaries, scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses', type=Path, required=True, help='JSONL output from llm.run')
    parser.add_argument('--items', type=Path, default=DATA / 'items.jsonl')
    parser.add_argument('--output', type=Path, default=Path('outputs/llm_scores'))
    args = parser.parse_args()
    summary, scores = rescore(load_items(args.items), read_jsonl(args.responses))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'scored.jsonl').write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in scores))
    labels = ['B', 'A', 'Q', 'Other']
    lines = ['# Recomputed LLM results', '',
             '| Language | Condition | Model | Completed | Correct B | Wrong A | Wrong Q | Other |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    with (args.output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['language', 'condition', 'model', 'n', *labels])
        for key, counts in sorted(summary.items()):
            language, group, model = key.split(':')
            writer.writerow([language, group, model, counts['n'], *[counts[k] for k in labels]])
            cells = [f'{100 * counts[k] / counts["n"]:.1f}% ({counts[k]})' for k in labels]
            lines.append('| ' + ' | '.join([language, group, model, str(counts['n']), *cells]) + ' |')
    (args.output / 'results.md').write_text('\n'.join(lines) + '\n')
    print(f'Rescored {len(scores)} responses across {len(summary)} conditions.')


if __name__ == '__main__':
    main()
