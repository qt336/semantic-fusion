"""Verify the public API benchmark files against the zero-token publication rule."""
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER_MODELS = {'kimi': 'Kimi K3', 'hy4': 'HY4 preview', 'deepseek': 'DeepSeek V4 Pro'}


# Bound the publication exceptions to these exact historical archives.
GPT5_HISTORICAL_RUNS = {'gpt5_minimal_full_20261125', 'gpt5_zenmux_20261125'}
MISSING_COUNT_ARCHIVES = {('../llm/', 'DeepSeek V4 Pro'), ('deepseek_v41_flash_20261020', 'deepseek-v4.1-flash-1017')}
THINKING_CONTROL_RUNS = {'deepseek_v41flash_thinking_20261124', 'deepseek_v4pro_thinking_20261124'}
HISTORICAL_ARCHIVES = {
    'results/extended/deepseek_v41flash_thinking_20261124/answers.jsonl.gz': 'be52fac604dae5110fad9fd5a9d10292e7e0172602a18706b5c70416a2f88343',
    'results/extended/deepseek_v4pro_thinking_20261124/answers.jsonl.gz': '7029463d5263b283cd29d33178d86a38816711972668703e7bd1e53360cb653b',
    'results/extended/deepseek_v41_flash_20261020/answers.jsonl.gz': '407edf412f928465c8457d86ed50457d0234c50b3a37d0ef85ee6a39e188171a',
    'results/extended/gpt5_minimal_full_20261125/answers.jsonl.gz': '3215e41d21340618efbebde10f6fd040a125bf34dc4b84d8e084248e86e87884',
    'results/extended/gpt5_zenmux_20261125/answers.jsonl.gz': '2579e7b45642da7643a4e5f70621a23cc10acb236b40cb4a669a5a8ae283edb9',
    'results/llm/deepseek.jsonl': 'fb876aa51ad358c116e20354b4ff1e58c28a1c8c01bed2614a08bd3018932903',
}


def historical(run, model):
    return (run in GPT5_HISTORICAL_RUNS and model == 'openai/gpt-5') or (run, model) in MISSING_COUNT_ARCHIVES or run in THINKING_CONTROL_RUNS


def read_csv(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


def verify():
    for name, digest in HISTORICAL_ARCHIVES.items():
        assert hashlib.sha256((ROOT.parents[1] / name).read_bytes()).hexdigest() == digest, name
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
                model = next(iter(models[run])) if len(models[run]) == 1 else row['model']
                missing_count = (run, model) in MISSING_COUNT_ARCHIVES
                if run in THINKING_CONTROL_RUNS:
                    identity = run, model, row['sample_id']
                    assert identity not in seen, identity
                    seen.add(identity)
                    assert type(row['reasoning_tokens']) is int and row['reasoning_tokens'] > 0, path
                    counts[run, model, row['language'], row['group']][row['score_on']] += 1
                    continue
                if missing_count:
                    assert row['model'] == row['model_returned'] == model, path
                    n = row['reasoning_token_count']
                elif run in GPT5_HISTORICAL_RUNS:
                    assert row['model'] == 'openai/gpt-5' and row['reasoning_item'] is True, path
                    n = row['reasoning_tokens']
                else:
                    assert row['has_reasoning_block'] is False and not row.get('reasoning_item'), path
                    n = row['reported_reasoning_tokens']
                assert (n is None if missing_count else type(n) is int and n == 0), path
                for field in ('reasoning_tokens', 'thinking_tokens', 'reasoning_token_count', 'thinking_token_count'):
                    if field in row:
                        assert type(row[field]) is type(n) and row[field] == n, (path, field)
                assert row.get('thinking_block_count', 0) == 0, path
                assert all(block == 'text' for block in row.get('blocks', [])), path
                assert not row.get('visible_reasoning') and not row.get('visible_reasoning_marker'), path
                assert row.get('eligible') is not False and row.get('status') in (None, 'completed'), path
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
                assert (n is None if slug == 'deepseek' else type(n) is int and n == 0)
                assert row['reasoning_char_count'] == 0
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
        assert c.total() == int(row['scored']), key
        missing = c.total() if key[:2] in MISSING_COUNT_ARCHIVES else 0
        positive = c.total() if key[0] in THINKING_CONTROL_RUNS else 0
        assert int(row['推理token未报告的题数']) == missing and int(row['推理token为正的题数']) == positive
        assert int(row['推理token为零的题数']) == c.total() - missing - positive
        assert int(row['historical_exception']) == (c.total() if historical(*key[:2]) else 0)
        kept[key[:2]] += c.total()
        received[key[:2]] += int(row['received'])
    assert set(counts) == groups
    audit = read_csv('eligibility.csv')
    assert len(audit) == len(kept) == len({(r['run'], r['model']) for r in audit})
    for row in audit:
        key = row['run'], row['model']
        assert key in kept and int(row['keep']) == kept[key]
        assert int(row['historical_exception']) == (kept[key] if historical(*key) else 0)
        assert int(row['received']) == received[key] == sum(int(row[k]) for k in
            ('keep', 'missing', 'positive', 'block', 'unverified', 'other'))
    return len(seen)


if __name__ == '__main__':
    print(f'{verify():,} eligible API experiment records verified (runs may overlap).')
