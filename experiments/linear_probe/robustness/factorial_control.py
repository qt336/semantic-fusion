from pathlib import Path
import json,time
from statistics import mean
import numpy as np
import torch
from model import ToyModel
from probe_common import SOURCE,read_special,PAIRS,dump
from driver import extract,predict
R=Path(__file__).parent

def run_seed(seed):
 with np.load(R/'datasets.npz') as z:
  used=set(map(tuple,z['large_labels'].tolist()))|set(map(tuple,z['val_labels'].tolist()))
 special=read_special(seed);rng=np.random.default_rng(20261112);groups={};payload={}
 for g in ['operator_0_1','operator_0_2','operator_2_1']:
  orig=special[g];A,B=map(int,orig['labels'][0]);alist=[a for a in range(3,331) if (a,B) not in used and (a,B) not in PAIRS];blist=[b for b in range(3,331) if (A,b) not in used and (A,b) not in PAIRS]
  newa=[];newb=[]
  for row in orig['x']:
   end=int(np.flatnonzero(row==332)[0]);mem=set(row[1:end].tolist());sa=set();sb=set()
   while len(sa)<8:
    a=int(rng.choice(alist));b=int(rng.choice(blist))
    if a==b or a in mem or b in mem or a in sa or b in sb or (a,b) in used or (a,b) in PAIRS:continue
    sa.add(a);sb.add(b);newa.append(a);newb.append(b)
  newa=np.array(newa);newb=np.array(newb);ix=np.arange(len(newa));bp=np.repeat(orig['bpos'],8);qp=np.repeat(orig['qpos'],8)
  for cell in ['replace_A','replace_B','replace_both']:
   x=np.repeat(orig['x'],8,axis=0);labels=np.repeat(orig['labels'],8,axis=0)
   if cell in ['replace_A','replace_both']:x[ix,bp-1]=newa;labels[:,0]=newa
   if cell in ['replace_B','replace_both']:x[ix,bp]=newb;x[ix,qp+1]=newb;labels[:,1]=newb
   assert all(tuple(pair) not in used and tuple(pair) not in PAIRS for pair in labels)
   groups[g+'/'+cell]={'x':x,'labels':labels,'bpos':bp,'qpos':qp}
   for k,v in groups[g+'/'+cell].items():payload[g+'/'+cell+'/'+k]=v
 np.savez_compressed(R/f'factorial_data_seed{seed}.npz',**payload)
 result=json.loads((R/f'results_seed{seed}.json').read_text());saved=torch.load(SOURCE/'runs'/f'heldout_pairs_seed{seed}'/'latest.pt',map_location='cuda',weights_only=False)
 model=ToyModel().cuda().eval();model.load_state_dict(saved['model']);model.requires_grad_(False)
 weights=torch.load(R/f'weights_seed{seed}.pt',map_location='cpu',weights_only=False)
 features={f'{cond}/{key}':extract(model,d,cond=='masked_layer0') for cond in ['baseline','masked_layer0'] for key,d in groups.items()}
 rows=[];preds={};contrasts={}
 for family,reader in result['selected'].items():
  w={k:v.cuda() for k,v in weights[reader].items()}
  with np.load(R/f'predictions_seed{seed}'/(reader+'.npz')) as savedpred:
   for cond in ['baseline','masked_layer0']:
    pergroup=[]
    for g in ['operator_0_1','operator_0_2','operator_2_1']:
     orig=special[g];basepred=savedpred[cond+'/'+g];base=(basepred==orig['labels'][:,1]).astype(float);scores={'semantic':float(base.mean())};percontext={}
     preds[family+'/'+cond+'/'+g+'/semantic']=basepred
     for cell in ['replace_A','replace_B','replace_both']:
      key=g+'/'+cell;dat=groups[key];f=features[cond+'/'+key]
      pred=predict((f-w['mean'])/w['sd'],w['w'],w['b']);correct=(pred==dat['labels'][:,1]).reshape(-1,8)
      scores[cell]=float(correct.mean());percontext[cell]=correct.mean(1);preds[family+'/'+cond+'/'+key]=pred
     delta=percontext['replace_A']+percontext['replace_B']-percontext['replace_both']-base
     rows.append({'seed':seed,'family':family,'reader':reader,'condition':cond,'group':g,'contexts':len(base),'alternatives_per_context':8,**scores,'interaction_pp':100*float(delta.mean())})
     pergroup.append(delta)
    contrasts[family+'/'+cond]=np.stack(pergroup).mean(0)
 np.savez_compressed(R/f'factorial_predictions_seed{seed}.npz',**preds)
 np.savez_compressed(R/f'factorial_context_contrasts_seed{seed}.npz',**contrasts)
 dump(R/f'factorial_results_seed{seed}.json',{'rows':rows,'definition':'delta = accuracy(replace_A)+accuracy(replace_B)-accuracy(replace_both)-accuracy(semantic)','all_counterfactual_pairs_absent_from_probe_training_and_validation':True})
 print('FACTORIAL COMPLETE',seed,flush=True)

def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
 for seed in [101,102,103]:
  if (R/f'factorial_results_seed{seed}.json').exists():continue
  while not (R/f'results_seed{seed}.json').exists():time.sleep(10)
  run_seed(seed);torch.cuda.empty_cache()
 print('ALL FACTORIAL COMPLETE',flush=True)
if __name__=='__main__':main()
