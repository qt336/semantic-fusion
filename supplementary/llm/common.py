"""Portable paths and the literal one-message benchmark prompts."""
from pathlib import Path
import json

DATA = Path(__file__).resolve().parent / 'data'


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_items(path=None):
    rows = read_jsonl(path or DATA / 'items.jsonl')
    result = {row['sample_id']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate sample IDs')
    return result


def make_prompt(a, b, c, language, group):
    separator = ', ' if group == 'comma_semantic' else (' ' if language == 'en' else '')
    display = separator.join([a, b, c])
    if language == 'en':
        return f'{display}. What word comes immediately before "{c}"? Output only one word.'
    if language == 'zh':
        return f'{display}。直接问“{c}”的前一个字是什么？只输出一个字即可。'
    raise ValueError(language)
