from pathlib import Path
import json,time
import numpy as np
import torch
from model import ToyModel
from probe_common import SOURCE,read_special,PAIRS,dump
from driver import extract,predict
R=Path(__file__).parent

def run_seed(seed):
 with np.load(R/'datasets.npz') as z:
  used=set(map(tuple,z['large_labels'].tolist()))|set(map(tuple,z['val_labels'].tolist()))
 special=read_special(seed);rng=np.random.default_rng(20261111);groups={}
 for group in ['operator_0_1','operator_0_2','operator_2_1']:
  original=special[group];xx=[];ll=[];bb=[];qq=[];indices=[]
  for i,(row,bp,qp,label) in enumerate(zip(original['x'],original['bpos'],original['qpos'],original['labels'])):
   a=int(label[0]);end=np.flatnonzero(row==332)[0];mem=row[1:end];options=[]
   for b in range(3,331):
    if b in mem or (a,b) in used or (a,b) in PAIRS:continue
    trial=row.copy();trial[bp]=b;trial[qp+1]=b
    if any(p in PAIRS for p in zip(trial[1:end-1],trial[2:end])):continue
    options.append(b)
   assert len(options)>=8
   for b in rng.choice(options,8,replace=False):
    trial=row.copy();trial[bp]=b;trial[qp+1]=b
    diff=np.flatnonzero(trial!=row);assert diff.tolist()==[int(bp),int(qp)+1]
    xx.append(trial);ll.append([a,b]);bb.append(bp);qq.append(qp);indices.append(i)
  groups[group.replace('operator_','matched_A8_')]={'x':np.array(xx),'labels':np.array(ll),'bpos':np.array(bb),'qpos':np.array(qq),'context_index':np.array(indices)}
 np.savez_compressed(R/f'matched_A8_seed{seed}.npz',**{f'{g}_{k}':v for g,d in groups.items() for k,v in d.items()})
 result=json.loads((R/f'results_seed{seed}.json').read_text());state=torch.load(SOURCE/'runs'/f'heldout_pairs_seed{seed}'/'latest.pt',map_location='cuda',weights_only=False)
 model=ToyModel().cuda().eval();model.load_state_dict(state['model']);model.requires_grad_(False)
 weights=torch.load(R/f'weights_seed{seed}.pt',map_location='cpu',weights_only=False)
 feats={f'{c}/{g}':extract(model,d,c=='masked_layer0') for c in ['baseline','masked_layer0'] for g,d in groups.items()}
 rows=[];preds={}
 for family,name in result['selected'].items():
  w={k:v.cuda() for k,v in weights[name].items()}
  for key,feat in feats.items():
   cond,g=key.split('/');pred=predict((feat-w['mean'])/w['sd'],w['w'],w['b']);labels=groups[g]['labels']
   rows.append({'seed':seed,'reader':name,'family':family,'condition':cond,'group':g,'n':len(pred),'correct':int((pred==labels[:,1]).sum()),'accuracy':float((pred==labels[:,1]).mean()),'reads_A':float((pred==labels[:,0]).mean())});preds[family+'/'+key]=pred
 np.savez_compressed(R/f'afixed_predictions_seed{seed}.npz',**preds)
 dump(R/f'afixed_results_seed{seed}.json',{'rows':rows,'held_out_from_probe_training_and_validation':True,'A_context_query_position_unchanged':True})
 print('A-FIXED COMPLETE',seed,flush=True)

def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
 for seed in [101,102,103]:
  if (R/f'afixed_results_seed{seed}.json').exists():continue
  while not (R/f'results_seed{seed}.json').exists():time.sleep(10)
  run_seed(seed);torch.cuda.empty_cache()
 print('ALL A-FIXED COMPLETE',flush=True)
if __name__=='__main__':main()
