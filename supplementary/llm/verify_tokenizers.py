"""Independently re-encode every prompt using pinned official tokenizers."""
import argparse
import hashlib
import json
from pathlib import Path

from .common import DATA, load_items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['deepseek', 'kimi', 'hy4'], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--local-dir', type=Path)
    source.add_argument('--download', action='store_true', help='Download pinned public tokenizer files (not model weights).')
    parser.add_argument('--cache-dir', type=Path, default=Path('outputs/tokenizers'))
    parser.add_argument('--output', type=Path, default=Path('outputs/tokenizer_check.json'))
    args = parser.parse_args()
    manifest = json.loads((DATA / 'tokenizer_manifest.json').read_text())[args.model]
    if args.download:
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(repo_id=manifest['repository'], revision=manifest['revision'],
                         allow_patterns=list(manifest['files_sha256']), cache_dir=str(args.cache_dir)))
    else:
        directory = args.local_dir
    for name, checksum in manifest['files_sha256'].items():
        if name == 'README.md':
            continue
        path = directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Tokenizer artifact checksum differs: {name}')
    if args.model == 'kimi':
        from transformers import AutoTokenizer
        # The pinned Kimi tokenizer requires its official custom Python tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=True, local_files_only=True)
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        decode = lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    else:
        # Load the official tokenizer JSON directly, without depending on a
        # model-specific class or downloading the model configuration/weights.
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
        decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    items = load_items()
    for row in items.values():
        ids = encode(row['display'])
        if decode(ids) != row['display']:
            raise ValueError(f'Tokenization round trip differs: {row["sample_id"]}')
        prompt_ids = encode(row['prompt'])
        if prompt_ids[:len(ids)] != ids:
            raise ValueError(f'Prompt boundary differs: {row["sample_id"]}')
        if row['group'] != 'comma_semantic':
            expected = [row['a'], row['b'], row['c']]
            if row['language'] == 'en':
                expected = [expected[0], ' ' + expected[1], ' ' + expected[2]]
            if len(ids) != 3 or [decode([i]) for i in ids] != expected:
                raise ValueError(f'Constituent alignment differs: {row["sample_id"]}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(model=args.model, revision=manifest['revision'],
                            prompts_checked=len(items), status='passed'), indent=2) + '\n')
    print(f'{args.model}: independently verified {len(items)} prompts.')


if __name__ == '__main__':
    main()
