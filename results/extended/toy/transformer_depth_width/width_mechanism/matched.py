"""Regenerate checkpoint comparisons from the two experiment directories."""
import csv,json,sys
from pathlib import Path
from statistics import mean

def read(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    return {r['epoch']:r['evaluation'] for r in rows if r.get('evaluation')}

rows=[]
for seed in (106,107,108):
    groups=[read(Path(base)/'runs'/f'transformer_seed{seed}'/'history.jsonl') for base in sys.argv[1:3]]
    for ep in sorted(set(groups[0]) & set(groups[1])):
        if ep > {106:250,107:350,108:350}[seed]: continue
        row={'seed':seed,'epoch':ep}
        for width,g in zip((256,1024),groups):
            m=g[ep]
            for name in ('arithmetic','standard','semantic','semantic_0','semantic_1','semantic_2'):
                row[f'w{width}_{name}']=100*m[name]['accuracy']
            for name,p in m['semantic']['rates'].items(): row[f'w{width}_{name}']=100*p
        rows.append(row)
out=Path(sys.argv[3])
with (out/'paired_epochs.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for threshold in (90,95,99):
    for seed in (106,107,108):
        subset=[r for r in rows if r['seed']==seed and min(r[f'w{w}_{k}'] for k in ('standard','arithmetic') for w in (256,1024))>=threshold]
        if subset: print(threshold,seed,[r['epoch'] for r in subset],{k:round(mean(r[f'w1024_{k}']-r[f'w256_{k}'] for r in subset),2) for k in ['correct','first','eos','other']})
print('matched',len(rows))
