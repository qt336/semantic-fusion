"""Verify the public API benchmark files against the zero-token publication rule."""
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER_MODELS = {'kimi': 'Kimi K3', 'hy4': 'HY4 preview', 'deepseek': 'DeepSeek V4 Pro'}


def read_csv(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


def verify():
    conditions = read_csv('conditions.csv')
    models = defaultdict(set)
    for row in conditions:
        models[row['run']].add(row['model'])
    counts = defaultdict(Counter)
    seen = set()

    for path in ROOT.rglob('answers.jsonl.gz'):
        run = path.parent.relative_to(ROOT).as_posix()
        assert run in models, path
        with gzip.open(path, 'rt') as stream:
            for row in map(json.loads, stream):
                n = row['reported_reasoning_tokens']
                assert type(n) is int and n == 0 and row['has_reasoning_block'] is False, path
                for field in ('reasoning_tokens', 'thinking_tokens', 'reasoning_token_count', 'thinking_token_count'):
                    if field in row:
                        assert type(row[field]) is int and row[field] == 0, (path, field)
                assert row.get('thinking_block_count', 0) == 0 and not row.get('reasoning_item'), path
                assert all(block == 'text' for block in row.get('blocks', [])), path
                assert not row.get('visible_reasoning') and not row.get('visible_reasoning_marker'), path
                assert row.get('eligible') is not False and row.get('status') in (None, 'completed'), path
                model = next(iter(models[run])) if len(models[run]) == 1 else row['model']
                identity = run, model, row['sample_id']
                assert identity not in seen, identity
                seen.add(identity)
                if 'group' in row:
                    group, outcome = row['group'], row['outcome']
                else:
                    group = 'position_' + str(row['position'])
                    outcome = 'B' if row['correct'] else 'Other'
                counts[run, model, row['language'], group][outcome] += 1

    for slug, model in PAPER_MODELS.items():
        with (ROOT.parent / 'llm' / (slug + '.jsonl')).open() as stream:
            for row in map(json.loads, stream):
                n = row['reasoning_token_count']
                assert type(n) is int and n == 0 and row['reasoning_char_count'] == 0
                identity = '../llm/', model, row['sample_id']
                assert identity not in seen
                seen.add(identity)
                label = next((label for label, field in [('B','b'), ('A','a'), ('Q','c')]
                              if row['first_answer'] == row[field]), 'Other')
                counts['../llm/', model, row['language'], row['group']][label] += 1

    kept, received = Counter(), Counter()
    groups = set()
    for row in conditions:
        key = tuple(row[k] for k in ('run', 'model', 'language', 'condition'))
        assert key not in groups, key
        groups.add(key)
        c = counts[key]
        assert [c[k] for k in ('B','A','Q','Other')] == [int(row[k]) for k in ('correct_b','wrong_a','wrong_c','other')], key
        assert c.total() == int(row['scored']) == int(row['推理token为零的题数']), key
        assert int(row['推理token为正的题数']) == int(row['推理token未报告的题数']) == 0
        kept[key[:2]] += c.total()
        received[key[:2]] += int(row['received'])
    assert set(counts) == groups
    audit = read_csv('eligibility.csv')
    assert len(audit) == len(kept) == len({(r['run'], r['model']) for r in audit})
    for row in audit:
        key = row['run'], row['model']
        assert key in kept and int(row['keep']) == kept[key]
        assert int(row['received']) == received[key] == sum(int(row[k]) for k in
            ('keep', 'missing', 'positive', 'block', 'unverified', 'other'))
    return len(seen)


if __name__ == '__main__':
    print(f'{verify():,} eligible API experiment records verified (runs may overlap).')
