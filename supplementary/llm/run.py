"""Repeat the paper's single-user-message API evaluation with reasoning disabled."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import requests

from .common import load_items, read_jsonl

PROVIDERS = {
    'kimi': dict(model='kimi-k3', key_env='KIMI_API_KEY', base_env='KIMI_BASE_URL',
                 base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', extra={'reasoning_effort': 'none'}),
    'deepseek': dict(model='deepseek-v4-pro', key_env='DEEPSEEK_API_KEY', base_env='DEEPSEEK_BASE_URL',
                     base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', extra={'enable_thinking': False}),
    'hy4': dict(model='hy4-preview', key_env='HY4_API_KEY', base_env='HY4_BASE_URL',
                base_url='https://tokenhub.tencentmaas.com/v1', extra={'thinking': {'type': 'disabled'}}),
}
RETRY_CODES = {408, 429, 500, 502, 503, 504}


def make_payload(model, item):
    config = PROVIDERS[model]
    return dict(model=config['model'], messages=[{'role': 'user', 'content': item['prompt']}],
                temperature=0, max_tokens=32, **config['extra'])


def run_one(model, item, key, base_url, attempts=5, timeout=180):
    identity = dict(model=model, sample_id=item['sample_id'])
    status_code = None
    for attempt in range(attempts):
        try:
            response = requests.post(base_url.rstrip('/') + '/chat/completions',
                                     headers={'Authorization': 'Bearer ' + key},
                                     json=make_payload(model, item), timeout=timeout)
            status_code = response.status_code
            if status_code in RETRY_CODES:
                if attempt + 1 < attempts:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                break
            if status_code != 200:
                # Do not save provider error bodies, request IDs or credentials.
                return dict(identity, status='request_failed', http_status=status_code)
            data = response.json()
            message = data['choices'][0]['message']
            answer = message.get('content')
            if answer is None:
                answer = ''
            if not isinstance(answer, str):
                return dict(identity, status='invalid_response', reason='non_text_content')
            reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
            usage = data.get('usage') or {}
            details = usage.get('completion_tokens_details') or {}
            reasoning_tokens = details.get('reasoning_tokens')
            row = dict(identity, answer_raw=answer, model_returned=data.get('model'),
                       reasoning_char_count=len(reasoning), reasoning_token_count=reasoning_tokens,
                       finish_reason=data['choices'][0].get('finish_reason'),
                       usage={k: usage[k] for k in ['prompt_tokens', 'completion_tokens', 'total_tokens'] if k in usage})
            if reasoning or (reasoning_tokens is not None and reasoning_tokens > 0):
                return dict(row, status='invalid_response', reason='reasoning_was_returned')
            return dict(row, status='completed')
        except requests.exceptions.JSONDecodeError:
            # Requests' JSONDecodeError is also a RequestException. Keep this
            # before transport errors so malformed successful replies are not retried.
            return dict(identity, status='invalid_response', reason='malformed_response')
        except requests.RequestException:
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 30))
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            # A malformed successful reply is retained as a protocol failure, not retried for a better answer.
            return dict(identity, status='invalid_response', reason='malformed_response')
    return dict(identity, status='request_failed', http_status=status_code, attempts=attempts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(PROVIDERS), required=True)
    parser.add_argument('--language', choices=['en', 'zh'])
    parser.add_argument('--group', choices=['semantic', 'random', 'comma_semantic'])
    parser.add_argument('--limit-per-condition', type=int, default=500)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--attempts', type=int, default=5)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Write request payloads without reading a key or contacting an API.')
    args = parser.parse_args()
    if min(args.limit_per_condition, args.workers, args.timeout, args.attempts) < 1:
        parser.error('Limits, workers, timeout and attempts must be positive')
    config = PROVIDERS[args.model]
    output = args.output or Path('outputs') / ('requests' if args.dry_run else 'responses') / (args.model + '.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    selected, counts = [], {}
    for item in load_items().values():
        if args.language and item['language'] != args.language:
            continue
        if args.group and item['group'] != args.group:
            continue
        group = (item['language'], item['group'])
        if counts.get(group, 0) < args.limit_per_condition:
            selected.append(item)
            counts[group] = counts.get(group, 0) + 1
    if args.dry_run:
        with output.open('w', encoding='utf-8') as stream:
            for item in selected:
                stream.write(json.dumps(dict(sample_id=item['sample_id'], request=make_payload(args.model, item)), ensure_ascii=False) + '\n')
        print(f'Wrote {len(selected)} payloads. No API requests were made.')
        return
    key = os.environ.get(config['key_env'])
    if not key:
        parser.error('Set ' + config['key_env'] + ' in your environment')
    base_url = os.environ.get(config['base_env'], config['base_url'])
    if not base_url.startswith('https://'):
        parser.error('API base URL must use HTTPS')
    completed = set()
    if output.exists():
        for row in read_jsonl(output):
            if row['model'] != args.model:
                parser.error('Output file contains a different model')
            if row.get('status') != 'request_failed':
                completed.add(row['sample_id'])
    pending = [item for item in selected if item['sample_id'] not in completed]
    # Remove old failed-attempt rows before retrying, so each item has a single final record.
    if output.exists():
        existing = [r for r in read_jsonl(output) if r.get('status') != 'request_failed']
        tmp = output.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in existing))
        tmp.replace(output)
    with output.open('a', encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, args.model, item, key, base_url, args.attempts, args.timeout) for item in pending]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print(f'{index}/{len(pending)} {row["sample_id"]}: {row["status"]}', flush=True)


if __name__ == '__main__':
    main()
