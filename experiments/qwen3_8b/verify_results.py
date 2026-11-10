"""Verify this result package using Python's standard library only."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parent


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_comma_results():
    report=[]
    current_source={r['sample_id']:r for r in read(ROOT/'early16_attention/items.jsonl') if r['group']=='semantic'}
    for name,total,review_count in [('comma_attention',2000,9),('legacy_comma',5000,8)]:
        folder=ROOT/name
        cases=read(folder/'items.jsonl');case_by={r['sample_id']:r for r in cases}
        assert len(cases)==len(case_by)==1000
        manifest=json.loads((folder/'manifest.json').read_text())
        assert hashlib.sha256((folder/'items.jsonl').read_bytes()).hexdigest()==manifest['input_sha256']
        validation=json.loads((folder/'final_validation.json').read_text())
        raw=[]
        for f in sorted((folder/'results').glob('worker_*.jsonl')):
            assert hashlib.sha256(f.read_bytes()).hexdigest()==validation['raw_result_sha256'][f.name]
            raw.extend(read(f))
        by={(r['sample_id'],r['condition']):r for r in raw}
        assert len(raw)==len(by)==total
        decisions=json.loads((folder/'scoring_review.json').read_text());assert len(decisions)==review_count
        final=read(folder/'scored_outputs.jsonl')
        assert len(final)==len({(r['sample_id'],r['condition']) for r in final})==total
        for r in final:
            original=by[(r['sample_id'],r['condition'])];case=case_by[r['sample_id']]
            assert r['answer_raw']==original['answer_raw'] and r['output_ids']==original['output_ids']
            assert all(r[k]==case[k] for k in ['language','group','a','b','c','display','prompt','constituent_positions'])
            decision=decisions.get(r['sample_id']+'|'+r['condition'])
            answer=original['extracted_answer']
            if decision:
                assert hashlib.sha256(original['answer_raw'].encode()).hexdigest()==decision['answer_raw_sha256']
                answer=decision['answer']
            assert r['extracted_answer']==answer
            assert r['outcome']==('B' if answer==r['b'] else 'A' if answer==r['a'] else 'Q' if answer==r['c'] else 'Other')
        summary=json.loads((folder/'summary.json').read_text())
        assert len(summary)==total//500
        for s in summary:
            counts=Counter(r['outcome'] for r in final if (r['language'],r['condition'])==(s['language'],s['condition']))
            assert sum(counts.values())==s['n']==500 and all(s[k]==counts[k] for k in ['B','A','Q','Other'])
        with (folder/'scored_outputs.csv').open(newline='') as f: csv_rows=list(csv.DictReader(f))
        assert len(csv_rows)==len(final)
        for a,b in zip(csv_rows,final):
            assert all(a[k]==b[k] for k in ['sample_id','condition','answer_raw','extracted_answer','outcome'])
        if name=='comma_attention':
            assert set(case_by)==set(current_source)
            for r in cases:
                source=current_source[r['sample_id']]
                assert all(r[k]==source[k] for k in ['language','a','b','c'])
                assert r['display']==', '.join([r['a'],r['b'],r['c']])
                assert r['prompt']==source['prompt'].replace(source['display'],r['display'],1)
            pairs=read(folder/'paired_changes.jsonl')
            assert len(pairs)==len({r['sample_id'] for r in pairs})==1000
            final_by={(r['sample_id'],r['condition']):r for r in final}
            for pair in pairs:
                for cond in ['baseline','early16']:
                    r=final_by[(pair['sample_id'],cond)]
                    assert pair[cond+'_raw']==r['answer_raw'] and pair[cond+'_outcome']==r['outcome']
            for r in final:
                assert r['masked_layers']==([] if r['condition']=='baseline' else list(range(16)))
            for s in json.loads((folder/'output_format.json').read_text()):
                rs=[r for r in final if r['language']==s['language'] and r['condition']==s['condition']]
                direct=sum((len(r['answer_raw'].strip())==1 if r['language']=='zh' else re.fullmatch(r"[A-Za-z]+(?:['’-][A-Za-z]+)*[.!?]?",r['answer_raw'].strip()) is not None) for r in rs)
                assert s['direct_single']==direct and s['answer_explanation']==len(rs)-direct
                assert not any(r['answer_raw'].strip().lower().startswith(r['display'].lower()) for r in rs)
        report.append(dict(experiment=name,generations=total,cells=len(summary),per_cell=500,reviewed=review_count))
    return report


def verify_fixed_prefix_results():
    report=[]
    for name,total,n_variants,n_review in [('fixed_prefix',8000,4,172),('fixed_prefix_exact_suffix',4000,2,79)]:
        folder=ROOT/name
        cases=read(folder/'items.jsonl');case_by={(r['sample_id'],r['variant']):r for r in cases}
        assert len(cases)==len(case_by)==total
        manifest=json.loads((folder/'manifest.json').read_text())
        assert hashlib.sha256((folder/'items.jsonl').read_bytes()).hexdigest()==manifest['input_sha256']
        ids={r['sample_id'] for r in cases};assert len(ids)==2000
        for sid in ids:
            group=[r for r in cases if r['sample_id']==sid]
            assert len(group)==n_variants
            for r in group:
                assert r['shared_prefix_ids']==group[0]['shared_prefix_ids']
                assert r['constituent_positions']==group[0]['constituent_positions']
                assert r['input_ids'][:r['shared_prefix_token_count']]==r['shared_prefix_ids']
        validation=json.loads((folder/'final_validation.json').read_text())
        raw=[]
        for f in sorted((folder/'results').glob('worker_*.jsonl')):
            assert hashlib.sha256(f.read_bytes()).hexdigest()==validation['raw_result_sha256'][f.name]
            raw.extend(read(f))
        raw_by={(r['sample_id'],r['variant']):r for r in raw}
        assert len(raw)==len(raw_by)==total
        decisions=json.loads((folder/'scoring_review.json').read_text());assert len(decisions)==n_review
        final=read(folder/'scored_outputs.jsonl');assert len(final)==total
        for r in final:
            raw=raw_by[(r['sample_id'],r['variant'])];case=case_by[(r['sample_id'],r['variant'])]
            assert r['answer_raw']==raw['answer_raw'] and r['output_ids']==raw['output_ids']
            assert r['prompt']==case['prompt'] and r['constituent_positions']==case['constituent_positions']
            answer=raw['extracted_answer'];d=decisions.get(r['sample_id']+'|'+r['variant'])
            if d:
                assert hashlib.sha256(r['answer_raw'].encode()).hexdigest()==d['answer_raw_sha256']
                answer=d['answer']
            assert r['extracted_answer']==answer
            assert r['outcome']==('B' if answer==r['b'] else 'A' if answer==r['a'] else 'Q' if answer==r['c'] else 'Other')
        summary=json.loads((folder/'summary.json').read_text());assert len(summary)==total//500
        for s in summary:
            c=Counter(r['outcome'] for r in final if (r['language'],r['group'],r['variant'])==(s['language'],s['group'],s['variant']))
            assert sum(c.values())==s['n']==500 and all(c[k]==s[k] for k in ['B','A','Q','Other'])
        audits=[]
        for f in sorted((folder/'results').glob('prefix_audit_*.jsonl')):
            assert hashlib.sha256(f.read_bytes()).hexdigest()==validation['prefix_audit_sha256'][f.name]
            audits.extend(read(f))
        assert len(audits)==len({r['sample_id'] for r in audits})==2000
        for r in audits:
            assert r['states_checked']==37 and len(r['max_abs_diff_by_variant'])==n_variants-1
            assert all(len(v)==37 for v in r['max_abs_diff_by_variant'].values())
            assert r['max_abs_diff']==max(x for v in r['max_abs_diff_by_variant'].values() for x in v)==0.0
            assert r['all_exact_equal']
        with (folder/'scored_outputs.csv').open(newline='') as f:csv_rows=list(csv.DictReader(f))
        assert len(csv_rows)==len(final)
        for a,b in zip(csv_rows,final):assert all(a[k]==b[k] for k in ['sample_id','variant','answer_raw','extracted_answer','outcome'])
        report.append(dict(experiment=name,generations=total,cells=len(summary),per_cell=500,reviewed=n_review,exact_prefix_state_matches=2000))
    previous={r['sample_id']:r for r in read(ROOT/'wording_comparison/items.jsonl') if r['variant']=='explicit_position'}
    for r in read(ROOT/'fixed_prefix_exact_suffix/items.jsonl'):
        if r['variant']=='exact_left':
            prefix='只看开头的这个序列：' if r['language']=='zh' else 'Read only this sequence: '
            assert previous[r['sample_id']]['prompt']==prefix+r['prompt']
    return report


def verify():
    report=[]
    inputs={}
    final_rows={}
    for name,key_field,expected,review_count in [
            ('wording_comparison','variant',6000,96),
            ('early16_attention','condition',4000,59)]:
        folder=ROOT/name
        manifest=json.loads((folder/'manifest.json').read_text())
        assert hashlib.sha256((folder/'items.jsonl').read_bytes()).hexdigest()==manifest['input_sha256']
        cases=read(folder/'items.jsonl')
        inputs[name]=cases
        case_by={(r['sample_id'],r['variant']):r for r in cases}
        assert len(case_by)==len(cases)==(6000 if key_field=='variant' else 2000)
        original_validation=json.loads((folder/'final_validation.json').read_text())
        rows=[]
        for f in sorted((folder/'results').glob('worker_*.jsonl')):
            assert hashlib.sha256(f.read_bytes()).hexdigest()==original_validation['raw_result_sha256'][f.name]
            rows.extend(read(f))
        raw_by={(r['sample_id'],r[key_field]):r for r in rows}
        assert len(rows)==len(raw_by)==expected
        decisions=json.loads((folder/'scoring_review.json').read_text())
        assert len(decisions)==review_count
        final=read(folder/'scored_outputs.jsonl')
        final_rows[name]=final
        assert len(final)==len({(r['sample_id'],r[key_field]) for r in final})==expected
        for r in final:
            raw=raw_by[(r['sample_id'],r[key_field])]
            assert r['answer_raw']==raw['answer_raw'] and r['output_ids']==raw['output_ids']
            case=case_by[(r['sample_id'],r['variant'])]
            for k in ['prompt','a','b','c','display','constituent_positions','three_aligned_tokens']:
                assert r[k]==case[k]
            d=decisions.get(r['sample_id']+'|'+r[key_field])
            answer=raw['extracted_answer']
            if d:
                assert hashlib.sha256(raw['answer_raw'].encode()).hexdigest()==d['answer_raw_sha256']
                answer=d['answer']
            assert r['extracted_answer']==answer
            outcome='B' if answer==r['b'] else 'A' if answer==r['a'] else 'Q' if answer==r['c'] else 'Other'
            assert r['outcome']==outcome
            if key_field=='condition':
                assert r['masked_layers']==([] if r['condition']=='baseline' else list(range(16)))
        summary=json.loads((folder/'summary.json').read_text())
        assert len(summary)==(12 if key_field=='variant' else 8)
        for s in summary:
            selected=[r for r in final if (r['language'],r['group'],r[key_field])==(s['language'],s['group'],s[key_field])]
            counts=Counter(r['outcome'] for r in selected)
            assert len(selected)==s['n']==500
            assert all(s[k]==counts[k] for k in ['B','A','Q','Other'])
        with (folder/'scored_outputs.csv').open(newline='') as f:
            csv_rows=list(csv.DictReader(f))
        assert len(csv_rows)==len(final)
        for a,b in zip(csv_rows,final):
            for k in ['sample_id','prompt','answer_raw','extracted_answer','outcome',key_field]:
                assert a[k]==b[k]
        with (folder/'summary.csv').open(newline='') as f:
            csv_summary=list(csv.DictReader(f))
        assert len(csv_summary)==len(summary)
        for a,b in zip(csv_summary,summary):
            assert all(int(a[k])==b[k] for k in ['n','B','A','Q','Other'])
            assert float(a['accuracy_pct'])==round(100*b['B']/b['n'],4)
        report.append(dict(experiment=name,generations=expected,cells=len(summary),per_cell=500,reviewed=review_count))
    assert inputs['early16_attention']==[r for r in inputs['wording_comparison'] if r['variant']=='explicit_position']
    pairs=read(ROOT/'early16_attention/paired_changes.jsonl')
    assert len(pairs)==len({r['sample_id'] for r in pairs})==2000
    index={(r['sample_id'],r['condition']):r for r in final_rows['early16_attention']}
    for pair in pairs:
        for condition in ['baseline','early16']:
            r=index[(pair['sample_id'],condition)]
            assert pair[condition+'_raw']==r['answer_raw']
            assert pair[condition+'_outcome']==r['outcome']
    format_counts=json.loads((ROOT/'output_format_counts.json').read_text())
    for s in format_counts:
        subset=[r for r in final_rows['early16_attention'] if r['language']==s['language'] and r['condition']==s['condition'] and (s['group']=='all' or r['group']==s['group'])]
        counts=Counter()
        for r in subset:
            raw=r['answer_raw'].strip()
            if (len(raw)==1 if r['language']=='zh' else re.fullmatch(r"[A-Za-z]+(?:['’-][A-Za-z]+)*[.!?]?",raw)):
                counts['single_character_or_word']+=1
            elif raw.lower()==r['a']+(' ' if r['language']=='en' else '')+r['b']:
                counts['AB_only']+=1
            elif raw.lower().startswith(r['display'].lower()):
                counts['ABC_echo_then_answer']+=1
            else:counts['other']+=1
        assert len(subset)==s['n']
        assert all(s[k]==counts[k] for k in ['single_character_or_word','AB_only','ABC_echo_then_answer','other'])
    report.extend(verify_comma_results())
    report.extend(verify_fixed_prefix_results())
    package_manifest=ROOT/'package_manifest.json'
    if package_manifest.exists():
        inventory=json.loads(package_manifest.read_text())['files']
        actual={f.relative_to(ROOT).as_posix() for f in ROOT.rglob('*') if f.is_file() and f!=package_manifest}
        assert set(inventory)==actual
        for name,entry in inventory.items():
            f=ROOT/name
            assert f.stat().st_size==entry['bytes']
            assert hashlib.sha256(f.read_bytes()).hexdigest()==entry['sha256']
    return dict(status='passed',total_generations=29000,experiments=report,attention_paired_items=3000,fixed_prefix_audit_records=4000,raw_outputs_preserved=True,output_format_counts_verified=True)


if __name__=='__main__':
    print(json.dumps(verify(),ensure_ascii=False,indent=2))
