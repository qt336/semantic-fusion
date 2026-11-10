from pathlib import Path
import json,hashlib
import numpy as np
from probe_common import read_special,PAIRS
R=Path(__file__).parent
seen=set();d={}
with np.load(R/'datasets.npz') as z:
 for split,reps in [('train',64),('val',8),('test',8),('extra',192),('fresh_test',32)]:
  v={k:z[f'{split}_{k}'] for k in ['x','labels','bpos','qpos']};d[split]=v
  for j in [0,1]:assert np.all(np.bincount(v['labels'][:,j],minlength=331)==reps)
  for row,labels,bp,qp in zip(v['x'],v['labels'],v['bpos'],v['qpos']):
   sig=tuple(row[:qp+1]);assert sig not in seen;seen.add(sig)
   assert row[bp]==labels[1] and row[bp-1]==labels[0]
   assert row[qp+1]==labels[1]
   sep=np.flatnonzero(row==332)[0]
   assert not any(pair in PAIRS for pair in zip(row[1:sep-1],row[2:sep]))
 large={k:z[f'large_{k}'] for k in ['x','labels','bpos','qpos']}
 for k in large:assert np.array_equal(large[k],np.concatenate([d['train'][k],d['extra'][k]]))
 used=set(map(tuple,large['labels'].tolist()))|set(map(tuple,d['val']['labels'].tolist()))
report={'disjoint_input_splits':True,'balanced_labels':True,'large_equals_train_plus_extra':True,'verified_seeds':[],'prediction_rows_verified':0}
for path in sorted(R.glob('results_seed*.json')):
 result=json.loads(path.read_text());seed=result['seed'];special=read_special(seed)
 groups={'ordinary_test':d['test'],'ordinary_fresh_test':d['fresh_test'],**special}
 with np.load(R/f'matched8_seed{seed}.npz') as z:
  for g in ['matched8_0_1','matched8_0_2','matched8_2_1']:
   v={k:z[f'{g}_{k}'] for k in ['x','labels','bpos','qpos','context_index']};groups[g]=v
   original=special[g.replace('matched8_','operator_')]
   assert len(v['x'])==8*len(original['x'])
   assert all(tuple(pair) not in used and tuple(pair) not in PAIRS for pair in v['labels'])
   for i,(row,bp,qi,idx) in enumerate(zip(v['x'],v['bpos'],v['qpos'],v['context_index'])):
    ref=original['x'][idx];diff=np.flatnonzero(row!=ref)
    assert diff.tolist()==[int(bp)-1]
    assert v['labels'][i,1]==original['labels'][idx,1]
   for i in range(len(original['x'])):
    assert len(set(v['labels'][i*8:(i+1)*8,0]))==8
 for reader in result['readers']:
  with np.load(R/f'predictions_seed{seed}'/(reader['name']+'.npz')) as p:
   for key,metrics in reader['results'].items():
    cond,g=key.split('/');pred=p[key];labels=groups[g]['labels']
    assert len(pred)==metrics['n']
    assert int((pred==labels[:,1]).sum())==metrics['correct']
    assert float((pred==labels[:,1]).mean())==metrics['accuracy']
    assert float((pred==labels[:,0]).mean())==metrics['reads_A']
    report['prediction_rows_verified']+=1
 report['verified_seeds'].append(seed)
report['randomized_matched_controls_held_out_from_probe_training_and_validation']=True
report['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'driver.py',R/'features.py',R/'model.py',R/'probe_common.py',R/'PROTOCOL.md',R/'datasets.npz']}
(R/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['verified_seeds'],report['prediction_rows_verified'],'prediction rows verified; all data checks passed')
