from pathlib import Path
import json
import numpy as np
from probe_common import read_special,PAIRS
R=Path(__file__).parent
with np.load(R/'datasets.npz') as z:used=set(map(tuple,z['large_labels'].tolist()))|set(map(tuple,z['val_labels'].tolist()))
report={'afixed_verified':[],'factorial_verified':[],'prediction_rows_verified':0}
for seed in [101,102,103]:
 special=read_special(seed)
 af=R/f'afixed_results_seed{seed}.json'
 if af.exists():
  rows=json.loads(af.read_text())['rows']
  with np.load(R/f'matched_A8_seed{seed}.npz') as data,np.load(R/f'afixed_predictions_seed{seed}.npz') as preds:
   for row in rows:
    g=row['group'];labels=data[g+'_labels'];p=preds[row['family']+'/'+row['condition']+'/'+g]
    assert (p==labels[:,1]).mean()==row['accuracy'];assert (p==labels[:,0]).mean()==row['reads_A']
    assert all(tuple(pair) not in used and tuple(pair) not in PAIRS for pair in labels)
    report['prediction_rows_verified']+=1
  report['afixed_verified'].append(seed)
 path=R/f'factorial_results_seed{seed}.json'
 if not path.exists():continue
 rows=json.loads(path.read_text())['rows']
 with np.load(R/f'factorial_data_seed{seed}.npz') as data,np.load(R/f'factorial_predictions_seed{seed}.npz') as preds,np.load(R/f'factorial_context_contrasts_seed{seed}.npz') as contrast:
  for g in ['operator_0_1','operator_0_2','operator_2_1']:
   orig=special[g];ref=np.repeat(orig['x'],8,axis=0);base_lab=np.repeat(orig['labels'],8,axis=0)
   bp=np.repeat(orig['bpos'],8);qp=np.repeat(orig['qpos'],8)
   for cell in ['replace_A','replace_B','replace_both']:
    prefix=g+'/'+cell+'/';x=data[prefix+'x'];labels=data[prefix+'labels']
    assert np.array_equal(data[prefix+'bpos'],bp) and np.array_equal(data[prefix+'qpos'],qp)
    assert all(tuple(pair) not in used and tuple(pair) not in PAIRS for pair in labels)
    for i in range(len(x)):
     expected=([int(bp[i])-1] if cell=='replace_A' else [int(bp[i]),int(qp[i])+1] if cell=='replace_B' else [int(bp[i])-1,int(bp[i]),int(qp[i])+1])
     assert np.flatnonzero(x[i]!=ref[i]).tolist()==expected
     assert x[i,bp[i]-1]==labels[i,0] and x[i,bp[i]]==labels[i,1]
     assert x[i,qp[i]+1]==labels[i,1]
    if cell=='replace_A':assert np.array_equal(labels[:,1],base_lab[:,1])
    if cell=='replace_B':assert np.array_equal(labels[:,0],base_lab[:,0])
   assert np.array_equal(data[g+'/replace_A/labels'][:,0],data[g+'/replace_both/labels'][:,0])
   assert np.array_equal(data[g+'/replace_B/labels'][:,1],data[g+'/replace_both/labels'][:,1])
  for row in rows:
   family=row['family'];cond=row['condition'];g=row['group'];base=preds[family+'/'+cond+'/'+g+'/semantic']==special[g]['labels'][:,1]
   assert base.mean()==row['semantic'];cc={}
   for cell in ['replace_A','replace_B','replace_both']:
    p=preds[family+'/'+cond+'/'+g+'/'+cell];labels=data[g+'/'+cell+'/labels']
    correct=(p==labels[:,1]).reshape(-1,8);assert correct.mean()==row[cell];cc[cell]=correct.mean(1)
   delta=cc['replace_A']+cc['replace_B']-cc['replace_both']-base
   assert abs(100*delta.mean()-row['interaction_pp'])<1e-10
   report['prediction_rows_verified']+=1
 report['factorial_verified'].append(seed)
(R/'controls_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
