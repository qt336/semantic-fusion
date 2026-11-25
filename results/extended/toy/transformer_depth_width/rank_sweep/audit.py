from pathlib import Path
import json
import numpy as np
import torch
root=Path(__file__).resolve().parent
source=Path('./source_L8_D1024/datasets/seed_106.npz')
data=np.load(source);checks=[]
torch.set_num_threads(3)
for width in (256,1024):
 dest=root/f'w{width}'
 if not (dest/'bases.pt').exists():continue
 bases=torch.load(dest/'bases.pt',weights_only=True)
 for layer,b in bases.items():
  for kind in ['b','arithmetic','random0','random1']:
   q=b[kind]
   assert float((q.T@q-torch.eye(q.shape[1])).abs().max())<1e-10
 baseline=np.load(dest/'baseline_predictions.npz')
 for path in sorted(dest.glob('layer*.json')):
  rec=json.loads(path.read_text())
  with np.load(path.with_suffix('.npz')) as predictions:
   for key in predictions.files:
    pred=predictions[key];gold=data[key+'_answer'];first=data[key+'_first']
    c={'correct':int((pred==gold).sum()),'first':int(((pred==first)&(pred!=gold)).sum()),'eos':int((pred==334).sum())}
    c['other']=len(pred)-sum(c.values())
    assert rec['metrics'][key]=={'n':len(pred),'counts':c}
    if rec['layer']==8:assert np.array_equal(pred,baseline[key])
  for group in ['arithmetic','semantic']:
   parts=[v for k,v in rec['metrics'].items() if k.startswith(group+'_')]
   assert rec['metrics'][group]=={'n':sum(v['n'] for v in parts),'counts':{k:sum(v['counts'][k] for v in parts) for k in ['correct','first','eos','other']}}
  if rec['variant']!='target':
   target=path.with_name(path.name.replace(rec['variant'],'target'))
   if target.exists():
    ref=json.loads(target.read_text())
    assert ref['perturbation']==rec['perturbation']
  checks.append({'width':width,'layer':rec['layer'],'kind':rec['kind'],'rank':rec['rank'],'variant':rec['variant'],'passed':True})
out={'verified_conditions':len(checks),'expected_conditions':540,'all_present':len(checks)==540,'checks':checks}
tmp=root/'audit.tmp';tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(root/'audit.json')
print('Audited',len(checks))
