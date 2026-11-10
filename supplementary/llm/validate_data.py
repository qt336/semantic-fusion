"""Check literal prompts, paired conditions and candidate pools."""
from collections import Counter
import json

from .common import DATA, load_items, make_prompt


def main():
    items = load_items()
    groups = Counter((r['language'], r['group']) for r in items.values())
    assert len(items) == 3000 and len(groups) == 6
    assert set(groups.values()) == {500}
    pools = json.loads((DATA / 'candidate_pools.json').read_text())
    assert len(set(pools['chinese_random_pool'])) == 189
    assert len(set(pools['english_random_pool'])) == 2694
    pairs = {'en': set(), 'zh': set()}
    for row in items.values():
        a, b, c = (row[k] for k in ['a', 'b', 'c'])
        assert len({a, b, c}) == 3 and row['gold'] == b
        assert row['prompt'] == make_prompt(a, b, c, row['language'], row['group'])
        if row['group'] == 'comma_semantic':
            original = items[row['paired_semantic_id']]
            assert [original[k] for k in ['a', 'b', 'c']] == [a, b, c]
        elif row['group'] == 'random':
            pool = pools['english_random_pool' if row['language'] == 'en' else 'chinese_random_pool']
            assert all(t in pool for t in [a, b, c])
        else:
            pairs[row['language']].add((a, b))
    assert len(pairs['en']) == 25 and len(pairs['zh']) == 118
    print('3,000 literal prompts, paired conditions and candidate pools pass.')
    print('Use llm.verify_tokenizers to check tokenization with the pinned official tokenizers.')


if __name__ == '__main__':
    main()
