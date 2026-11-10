from pathlib import Path
import json,hashlib,time,math
import numpy as np
import torch
from torch.nn import functional as F
from model import ToyModel,rope
from probe_common import SOURCE,normalize,ridge_fit,ridge_predict,softmax_fit,softmax_predict,read_special,dump
ROOT=Path(__file__).parent
COMPONENTS=['embedding','attention_only','embedding_plus_attention','block1_full','block1_minus_embedding','attention_mlp_no_embedding']

@torch.no_grad()
def components(model,x,bpos,masked=False,audit=False):
 n,t=x.shape;ix=torch.arange(n,device=x.device)
 emb=model.embedding(x);block=model.blocks[0];mix=block.mixer;z=block.ln1(emb)
 q,k,v=mix.qkv(z).reshape(n,t,3,mix.heads,mix.width//mix.heads).unbind(2)
 q,k,v=[a.transpose(1,2) for a in (q,k,v)];q,k=rope(q),rope(k)
 mask=None
 if masked:
  mask=torch.ones((n,1,t,t),dtype=torch.bool,device=x.device).tril()
  mask[ix,0,bpos,bpos-1]=False
 raw=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=mask is None)
 branch=mix.out(raw.transpose(1,2).reshape(n,t,mix.width))
 full=emb+branch;full=full+block.mlp(block.ln2(full))
 no_embedding=branch+block.mlp(block.ln2(branch))
 # Decompose only the displayed B query's attention by source. Constant output bias
 # belongs to the whole branch, and is not assigned to an individual source.
 qb=q[ix,:,bpos,:]
 logits=torch.einsum('nhd,nhtd->nht',qb,k)/math.sqrt(q.shape[-1])
 allowed=torch.arange(t,device=x.device)[None,:]<=bpos[:,None]
 if masked: allowed[ix,bpos-1]=False
 logits=logits.masked_fill(~allowed[:,None,:],float('-inf'))
 alpha=logits.softmax(-1)
 aa=alpha[ix,:,bpos-1];ab=alpha[ix,:,bpos]
 va=v[ix,:,bpos-1,:];vb=v[ix,:,bpos,:]
 from_a=F.linear((aa[:,:,None]*va).reshape(n,-1),mix.out.weight,None)
 from_b=F.linear((ab[:,:,None]*vb).reshape(n,-1),mix.out.weight,None)
 if audit:
  expected=branch[ix,bpos]+block.mlp(block.ln2(branch[ix,bpos]))
  torch.testing.assert_close(no_embedding[ix,bpos],expected,rtol=1e-5,atol=1e-5)
  direct=F.linear(torch.einsum('nht,nhtd->nhd',alpha,v).reshape(n,-1),mix.out.weight,mix.out.bias)
  torch.testing.assert_close(direct,branch[ix,bpos],rtol=1e-5,atol=1e-5)
  if not masked:
   assert torch.equal(branch,block.mixer(block.ln1(emb)))
   assert torch.equal(full,block(emb))
  else: assert torch.all(aa==0) and torch.all(from_a==0)
 f=torch.stack([emb[ix,bpos],branch[ix,bpos],(emb+branch)[ix,bpos],full[ix,bpos],(full-emb)[ix,bpos],no_embedding[ix,bpos]],1)
 return f,torch.stack([aa.mean(1),ab.mean(1),1-aa.mean(1)-ab.mean(1)],1)

@torch.no_grad()
def extract(model,data,masked,device):
 feats=[];att=[]
 for start in range(0,len(data['x']),512):
  sl=slice(start,start+512)
  f,a=components(model,torch.as_tensor(data['x'][sl],device=device),torch.as_tensor(data['bpos'][sl],device=device),masked)
  feats.append(f);att.append(a.cpu().numpy())
 return torch.cat(feats),np.concatenate(att)

def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');device=torch.device('cuda')
 with np.load(ROOT/'probe_data.npz') as z:
  data={split:{k:z[f'{split}_{k}'] for k in ['x','labels','bpos','qpos']} for split in ['train','val','test']}
 results=[];attention=[];audits={}
 for seed in [101,102,103]:
  print('START',seed,flush=True);start=time.monotonic()
  path=SOURCE/'runs'/f'heldout_pairs_seed{seed}'/'latest.pt'
  saved=torch.load(path,map_location=device,weights_only=False);assert saved['epoch']==1200
  model=ToyModel().to(device).eval();model.load_state_dict(saved['model']);model.requires_grad_(False)
  x=torch.as_tensor(data['test']['x'][:32],device=device);bp=torch.as_tensor(data['test']['bpos'][:32],device=device)
  for masked in [False,True]:
   f,_=components(model,x,bp,masked,audit=True)
   changed=x.clone()
   for i in range(len(x)): changed[i,bp[i]+1:]=17
   cf,_=components(model,changed,bp,masked)
   assert torch.equal(f,cf),'Future leakage'
  audits[str(seed)]={'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'epoch':1200,'attention_branch_matches_original':True,'block_output_matches_original':True,'source_sum_matches_attention_output':True,'masked_A_contribution_exact_zero':True,'no_future_leakage':True,'no_embedding_expression_checked':True}
  tr,_=extract(model,data['train'],False,device);va,_=extract(model,data['val'],False,device)
  mean=tr.mean(0,keepdim=True);sd=tr.std(0,keepdim=True).clamp_min(1e-6)
  tr=normalize(tr,mean,sd);va=normalize(va,mean,sd)
  y=torch.as_tensor(data['train']['labels'],device=device);vy=torch.as_tensor(data['val']['labels'],device=device)
  rw,rm=ridge_fit(tr,y,va,vy);sw,sm=softmax_fit(tr,y,va,vy,100)
  torch.save({'components':COMPONENTS,'mean':mean.cpu(),'std':sd.cpu(),'ridge':rw.cpu(),'softmax_w':sw[0].cpu(),'softmax_b':sw[1].cpu()},ROOT/f'probes_seed{seed}.pt')
  dump(ROOT/f'fit_seed{seed}.json',{'components':COMPONENTS,'ridge':rm,'softmax':sm})
  del tr,va
  groups={'ordinary_test':data['test'],**read_special(seed)};predictions={}
  for masked in [False,True]:
   cond='masked_layer0' if masked else 'baseline'
   for name,dat in groups.items():
    feat,aa=extract(model,dat,masked,device);feat=normalize(feat,mean,sd)
    attention.append({'seed':seed,'group':name,'condition':cond,'mean_attention_A':float(aa[:,0].mean()),'mean_attention_self':float(aa[:,1].mean()),'mean_attention_other':float(aa[:,2].mean())})
    for family,w,fun in [('ridge',rw,ridge_predict),('softmax',sw,softmax_predict)]:
     pred=fun(w,feat);predictions[f'{cond}_{name}_{family}']=pred
     for i,component in enumerate(COMPONENTS):
      for j,target in enumerate('AB'):
       results.append({'seed':seed,'group':name,'condition':cond,'family':family,'component':component,'target':target,'n':len(pred),'accuracy':float((pred[:,i,j]==dat['labels'][:,j]).mean())})
  np.savez_compressed(ROOT/f'predictions_seed{seed}.npz',**predictions)
  dump(ROOT/'results.json',{'probe':results,'attention':attention,'audits':audits,'completed_seeds':list(audits)})
  print('DONE',seed,'seconds',time.monotonic()-start,flush=True)
  del model,rw,sw;torch.cuda.empty_cache()
 dump(ROOT/'status.json',{'state':'complete','seeds':[101,102,103]})
 print('COMPLETE',flush=True)

if __name__=='__main__':main()
