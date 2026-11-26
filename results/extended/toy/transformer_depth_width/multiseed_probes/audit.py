"""Recount saved new probe predictions; historical seed106 is sourced explicitly."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
verified=[]
for seed in [107,108]:
    for width in [256,1024]:
        folder=ROOT/f'seed{seed}_w{width}'
        for path in sorted(folder.glob('layer??.json')):
            rec=json.loads(path.read_text())
            with np.load(path.with_name(path.stem+'_predictions.npz')) as d:
                for key in ['test','semantic_0','semantic_1','semantic_2']:
                    assert len(d[key])==rec[key]['n']
                    assert int((d[key]==d[key+'_gold']).sum())==rec[key]['correct']
                assert rec['semantic']['correct']==sum(rec[f'semantic_{i}']['correct'] for i in range(3))
            verified.append([seed,width,rec['layer']])
record={'new_layers_verified':len(verified),'expected_new_layers':32,'verified':verified,
        'seed106':'Reused historical baseline counts in subspace_intervention/probe_projection_effect.json',
        'missing_pairs':[109,110],'underqualified_pair':[108]}
(ROOT/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
print(record['new_layers_verified'],'/32 new layers verified')
