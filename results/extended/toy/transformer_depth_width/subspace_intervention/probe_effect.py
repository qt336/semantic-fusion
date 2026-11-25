"""Measure how much the chosen projection changes the original identity reader."""
from pathlib import Path
import json,sys
import numpy as np
import torch
from torch.nn import functional as F
from run import features,BASE,ROOT,ToyModel,atomic

width=int(sys.argv[1]);dest=ROOT/f'w{width}'
ck=torch.load(dest/'backbone.pt',map_location='cpu',weights_only=True,mmap=True)
model=ToyModel('transformer',width=width,heads=8,layers=8).cuda().eval().requires_grad_(False)
model.load_state_dict(ck['model']);del ck
ordinary=np.load(BASE/'all_layers_probe_seed106/ordinary.npz')
data=np.load(BASE/'datasets/seed_106.npz')
groups={'ordinary':(ordinary['test_x'],ordinary['test_bpos'],ordinary['test_labels'][:,1])}
for i in range(3):groups[f'semantic_{i}']=(data[f'semantic_{i}_x'],data[f'semantic_{i}_edge'][:,0],data[f'semantic_{i}_answer'])
results={}
torch.set_num_threads(3);torch.set_float32_matmul_precision('high')
with torch.no_grad():
 for name,(x,bp,gold) in groups.items():
  x=x.copy();x[np.arange(x.shape[1])[None,:]>bp[:,None]]=335
  feats=features(model,x,bp)
  for layer in range(8):
   state=torch.load(dest/f'layer{layer+1:02}_directions.pt',weights_only=True)
   reader=state['reader'];q=state['b_basis'];u=feats[layer]
   pair={}
   for tag,z in [('baseline',u),('removed_b',u-(u@q)@q.T)]:
    pred=F.linear((z-reader['mean'])/reader['sd'],reader['weight'],reader['bias']).argmax(1).numpy()
    pair[tag]={'n':len(pred),'correct':int((pred==gold).sum())}
   weight=(reader['weight']-reader['weight'].mean(0))/reader['sd'][None,:]
   pair['readout_weight_energy_fraction']=float(((weight@q)**2).sum()/(weight**2).sum())
   results.setdefault(str(layer+1),{})[name]=pair
atomic(dest/'probe_projection_effect.json',results)
print('Probe projection effect saved',width)
