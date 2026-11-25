"""Verify saved predictions and projection geometry without refitting."""
from pathlib import Path
import json
import numpy as np
import torch
root=Path(__file__).resolve().parent
base=Path('./source_L8_D1024')
d=np.load(base/'datasets/seed_106.npz')
checks=[]
for folder in sorted(root.glob('w[0-9]*')):
    if not folder.is_dir():continue
    for path in sorted(folder.glob('layer*.json')):
        row=json.loads(path.read_text())
        if not row.get('complete') or not row.get('double_orthogonal'):continue
        layer=row['layer']
        state=torch.load(folder/f'layer{layer:02}_directions.pt',weights_only=True)
        b=state['b_basis'].double()
        assert torch.max(torch.abs(b.T@b-torch.eye(16)))<2e-5
        a=state.get('arithmetic_basis')
        overlap=None
        if a is not None:
            a=a.double()
            assert torch.max(torch.abs(a.T@a-torch.eye(16)))<2e-5
            overlap=float((b.T@a).abs().max())
            assert overlap<2e-4,overlap
        for cond,m in row['conditions'].items():
            pred=np.load(folder/f'layer{layer:02}_{cond}_predictions.npz')
            for key in pred.files:
                p=pred[key];g=d[key+'_answer'];f=d[key+'_first']
                got={'n':len(p),'correct':int((p==g).sum()),'first':int(((p==f)&(p!=g)).sum()),'eos':int((p==334).sum())}
                assert all(m[key][k]==v for k,v in got.items())
        if layer==8:
            base=np.load(folder/'baseline_predictions.npz')
            for cond in row['conditions']:
                pred=np.load(folder/f'layer{layer:02}_{cond}_predictions.npz')
                assert all(np.array_equal(base[k],pred[k]) for k in base.files)
        checks.append({'width':row['width'],'layer':layer,'max_cross_basis_dot':overlap,'passed':True})
(root/'audit.json').write_text(json.dumps(checks,indent=2)+'\n')
print('Verified layers',len(checks))
