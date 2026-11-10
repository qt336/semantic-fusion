from pathlib import Path
import argparse,json,time,hashlib,sys,csv
import numpy as np
import torch
from torch.nn import functional as F
from model import ToyModel,rope
ROOT=Path(__file__).parent
SOURCE=(Path(__file__).resolve().parents[3] / 'experiments/transformer_controls')
P=331; BOS=333;SEP=332;Q=331;EOS=334;PAD=335
PAIRS=[(0,1),(0,2),(2,1),(9,10),(15,20)]
EXCLUDED=set(PAIRS)
STAGES=['embedding','block1','block2','block3','block4','final_norm']

def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def generated_data():
 rng=np.random.default_rng(20261107); seen=set(); result={}
 for split,reps in [('train',64),('val',8),('test',8)]:
  xx=[]; yy=[]; bp=[]; qp=[]
  for _ in range(reps):
   while True:
    bs=rng.permutation(P)
    if all(a!=int(b) and (a,int(b)) not in EXCLUDED for a,b in enumerate(bs)):break
   for a,b in enumerate(bs):
    b=int(b)
    while True:
     n=int(rng.integers(3,7)); start=int(rng.integers(n-2))
     pool=[i for i in range(P) if i not in [a,b]]
     other=rng.choice(pool,n-2,replace=False).tolist()
     mem=other[:start]+[a,b]+other[start:]; query=mem[start+2]
     sig=(*mem,-1,start+2)
     if sig in seen or any(pair in EXCLUDED for pair in zip(mem,mem[1:])) or (query,b) in EXCLUDED:continue
     seen.add(sig);break
    row=[BOS,*mem,SEP,Q,query,b,EOS]
    xx.append(row+[PAD]*(12-len(row)));yy.append([a,b]);bp.append(start+2);qp.append(n+3)
  result[split]={'x':np.array(xx,dtype=np.int64)[:,:-1], 'labels':np.array(yy), 'bpos':np.array(bp), 'qpos':np.array(qp)}
  for target in range(2):assert np.all(np.bincount(result[split]['labels'][:,target],minlength=P)==reps)
 return result

@torch.no_grad()
def traced(model,x,bpos,masked=False):
 batch,t=x.shape; indices=torch.arange(batch,device=x.device)
 h=model.embedding(x);features=[h[indices,bpos]]
 for li,block in enumerate(model.blocks):
  z=block.ln1(h); mixer=block.mixer
  q,k,v=mixer.qkv(z).view(batch,t,3,mixer.heads,mixer.width//mixer.heads).unbind(2)
  q,k,v=(u.transpose(1,2) for u in (q,k,v))
  mask=None
  if masked and li==0:
   mask=torch.ones((batch,1,t,t),dtype=torch.bool,device=x.device).tril()
   mask[indices,0,bpos,bpos-1]=False
   assert not mask[indices,0,bpos,bpos-1].any()
  out=F.scaled_dot_product_attention(rope(q),rope(k),v,attn_mask=mask,is_causal=mask is None)
  h=h+mixer.out(out.transpose(1,2).reshape(batch,t,mixer.width))
  h=h+block.mlp(block.ln2(h));features.append(h[indices,bpos])
 h=model.norm(h);features.append(h[indices,bpos])
 return torch.stack(features,1),model.head(h)

@torch.no_grad()
def extract(model,data,masked,device):
 ff=[];pp=[]
 for start in range(0,len(data['x']),512):
  sl=slice(start,start+512)
  x=torch.as_tensor(data['x'][sl],device=device);bp=torch.as_tensor(data['bpos'][sl],device=device)
  features,logits=traced(model,x,bp,masked)
  qp=torch.as_tensor(data['qpos'][sl],device=device)
  ff.append(features);pp.append(logits[torch.arange(len(x),device=device),qp].argmax(-1).cpu().numpy())
 return torch.cat(ff),np.concatenate(pp)

def read_special(seed):
 sets={}
 with np.load(SOURCE/'datasets'/f'seed{seed}.npz') as z:
  for a,b in PAIRS:
   key=f'{"operator" if (a,b) in PAIRS[:3] else "ordinary"}_{a}_{b}'
   x=z[key+'_x']; hit=(x[:,:-1]==a)&(x[:,1:]==b)
   assert np.all(hit.sum(1)==1)
   bpos=hit.argmax(1)+1; qpos=z[key+'_pos']; labels=np.tile([a,b],(len(x),1))
   sets[key]={'x':x,'labels':labels,'bpos':bpos,'qpos':qpos}
   if (a,b) in PAIRS[:3]:
    xx=x.copy()
    for i in range(len(xx)):
     mem=xx[i,1:np.flatnonzero(xx[i]==SEP)[0]].tolist()
     # Preserve B, query, position, length, all other symbols; ordinary A from unused IDs.
     for replacement in range(3,P):
      if replacement in mem or (replacement,b) in EXCLUDED: continue
      trial=xx[i].copy();trial[bpos[i]-1]=replacement
      end=int(np.flatnonzero(trial==SEP)[0])
      if any(p in EXCLUDED for p in zip(trial[1:end-1],trial[2:end])):continue
      xx[i]=trial;break
     else:raise AssertionError('No matched control')
    ll=labels.copy();ll[:,0]=xx[np.arange(len(xx)),bpos-1]
    sets[f'matched_B_{a}_{b}']={'x':xx,'labels':ll,'bpos':bpos.copy(),'qpos':qpos.copy()}
 return sets

def normalize(features,mean,sd):return (features-mean)/sd

@torch.no_grad()
def ridge_fit(x,y,v,vy):
 # Heads ordered by stage, then A/B. Shared standardization is learned only from x.
 xt=x.permute(1,0,2).double(); vv=v.permute(1,0,2).double()
 xt=torch.cat([xt,torch.ones_like(xt[:,:,:1])],-1);vv=torch.cat([vv,torch.ones_like(vv[:,:,:1])],-1)
 target=torch.cat([F.one_hot(y[:,j],P) for j in range(2)],-1).double()
 gram=xt.transpose(1,2)@xt/len(y);rhs=xt.transpose(1,2)@target/len(y)
 eye=torch.eye(xt.shape[-1],device=xt.device,dtype=xt.dtype);eye[-1,-1]=0
 best=torch.full((6,2),-1.,device=xt.device);weights=torch.zeros((6,257,2*P),device=xt.device,dtype=xt.dtype);selected=torch.zeros_like(best)
 for alpha in [1e-6,1e-4,1e-2,1.0]:
  w=torch.linalg.solve(gram+alpha*eye,rhs)
  scores=(vv@w).reshape(6,len(vy),2,P)
  acc=(scores.argmax(-1)==vy[None]).float().mean(1)
  for l in range(6):
   for j in range(2):
    if acc[l,j]>best[l,j]:
     best[l,j]=acc[l,j];selected[l,j]=alpha;weights[l,:,j*P:(j+1)*P]=w[l,:,j*P:(j+1)*P]
 return weights.float(),{'val_accuracy':best.cpu().tolist(),'selected_alpha':selected.cpu().tolist()}

@torch.no_grad()
def ridge_predict(w,x):
 z=torch.cat([x.permute(1,0,2),torch.ones_like(x.permute(1,0,2)[:,:,:1])],-1)
 return (z@w).reshape(6,len(x),2,P).argmax(-1).permute(1,0,2).cpu().numpy()

def softmax_fit(x,y,v,vy,epochs):
 # Twelve independent affine heads: no nonlinear transformation of features.
 torch.manual_seed(711)
 w=torch.nn.Parameter(torch.zeros(6,256,2*P,device=x.device));b=torch.nn.Parameter(torch.zeros(6,1,2*P,device=x.device))
 opt=torch.optim.AdamW([{'params':[w],'weight_decay':.01},{'params':[b],'weight_decay':0.}],lr=.01)
 best=torch.full((6,2),-1.,device=x.device);bestw=w.detach().clone();bestb=b.detach().clone();selected=torch.zeros_like(best)
 gen=torch.Generator(device=x.device).manual_seed(812)
 for epoch in range(1,epochs+1):
  order=torch.randperm(len(x),device=x.device,generator=gen)
  for offset in range(0,len(x),2048):
   ids=order[offset:offset+2048];z=x[ids].permute(1,0,2)
   scores=(z@w+b).reshape(6,len(ids),2,P)
   target=y[ids][None].expand(6,-1,-1)
   loss=F.cross_entropy(scores.reshape(-1,P),target.reshape(-1))
   opt.zero_grad();loss.backward();opt.step()
  if epoch==1 or epoch%5==0:
   with torch.no_grad():
    scores=(v.permute(1,0,2)@w+b).reshape(6,len(v),2,P)
    acc=(scores.argmax(-1)==vy[None]).float().mean(1)
    for l in range(6):
     for j in range(2):
      if acc[l,j]>best[l,j]:
       best[l,j]=acc[l,j];selected[l,j]=epoch
       bestw[l,:,j*P:(j+1)*P]=w[l,:,j*P:(j+1)*P];bestb[l,:,j*P:(j+1)*P]=b[l,:,j*P:(j+1)*P]
  if epoch%25==0: print('softmax epoch',epoch,'val final B',round(best[-1,1].item(),4),flush=True)
 return (bestw,bestb),{'val_accuracy':best.cpu().tolist(),'selected_epoch':selected.cpu().tolist(),'epochs':epochs}

@torch.no_grad()
def softmax_predict(wb,x):
 w,b=wb
 return (x.permute(1,0,2)@w+b).reshape(6,len(x),2,P).argmax(-1).permute(1,0,2).cpu().numpy()

def main(args):
 torch.set_num_threads(2);device=torch.device('cuda');torch.set_float32_matmul_precision('highest')
 data=generated_data()
 np.savez_compressed(ROOT/'probe_data.npz',**{f'{s}_{k}':v for s,d in data.items() for k,v in d.items()})
 dump(ROOT/'data_audit.json',{'counts':{s:len(d['x']) for s,d in data.items()},'balanced_all_331_labels':True,'unique_contexts_across_splits':True,'excluded_pairs':PAIRS,'chance_percent':100/P})
 outputs=[];behavior=[];audits={}
 for seed in args.seeds:
  print('START seed',seed,flush=True); start=time.monotonic()
  model=ToyModel().to(device).eval();ckpt=SOURCE/'runs'/f'heldout_pairs_seed{seed}'/'latest.pt'
  saved=torch.load(ckpt,map_location=device,weights_only=False);assert saved['epoch']==1200
  model.load_state_dict(saved['model']);model.requires_grad_(False)
  sample=torch.as_tensor(data['test']['x'][:16],device=device);bp=torch.as_tensor(data['test']['bpos'][:16],device=device)
  with torch.no_grad():
   f,logits=traced(model,sample,bp);ref=model(sample)
   assert torch.equal(ref,logits),'Empty intervention differs'
   changed=sample.clone()
   for i in range(len(changed)): changed[i,bp[i]+1:]=27
   cf,_=traced(model,changed,bp)
   assert torch.equal(f,cf),'B feature leaks future tokens'
   # At layer zero removing A must leave the embedding unchanged.
   mf,_=traced(model,sample,bp,True);assert torch.equal(f[:,0],mf[:,0])
  audits[str(seed)]={'checkpoint':str(ckpt),'sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'epoch':1200,'unmodified_logits_exact':True,'no_future_feature_leak':True,'masked_embedding_unchanged':True}
  train,_=extract(model,data['train'],False,device);val,_=extract(model,data['val'],False,device)
  mean=train.mean(0,keepdim=True);sd=train.std(0,keepdim=True).clamp_min(1e-6)
  train=normalize(train,mean,sd);val=normalize(val,mean,sd)
  y=torch.as_tensor(data['train']['labels'],device=device);vy=torch.as_tensor(data['val']['labels'],device=device)
  rw,ra=ridge_fit(train,y,val,vy);print('ridge fitted',seed,flush=True)
  sw,sa=softmax_fit(train,y,val,vy,args.probe_epochs)
  torch.save({'mean':mean.cpu(),'std':sd.cpu(),'ridge':rw.cpu(),'softmax_w':sw[0].cpu(),'softmax_b':sw[1].cpu(),'ridge_selection':ra,'softmax_selection':sa},ROOT/f'probes_seed{seed}.pt')
  dump(ROOT/f'fit_seed{seed}.json',{'ridge':ra,'softmax':sa})
  del train,val
  sets={'ordinary_test':data['test'],**read_special(seed)}
  predictions={}
  for masked in [False,True]:
   condition='masked_layer0' if masked else 'baseline'
   for name,d in sets.items():
    feat,pred=extract(model,d,masked,device);feat=normalize(feat,mean,sd);labels=d['labels']
    behavior.append({'seed':seed,'condition':condition,'group':name,'n':len(pred),'correct':float((pred==labels[:,1]).mean()),'first':float((pred==labels[:,0]).mean())})
    predictions[f'{condition}_{name}_model']=pred
    for family,w,func in [('ridge',rw,ridge_predict),('softmax',sw,softmax_predict)]:
     pr=func(w,feat);predictions[f'{condition}_{name}_{family}']=pr
     for li,stage in enumerate(STAGES):
      for target in range(2):
       outputs.append({'seed':seed,'condition':condition,'group':name,'family':family,'stage':stage,'target':'AB'[target],'n':len(labels),'accuracy':float((pr[:,li,target]==labels[:,target]).mean())})
  np.savez_compressed(ROOT/f'predictions_seed{seed}.npz',**predictions)
  dump(ROOT/'results.json',{'probe':outputs,'behavior':behavior,'audits':audits,'completed_seeds':list(audits)})
  print('DONE seed',seed,'seconds',time.monotonic()-start,flush=True)
  del model,rw,sw;torch.cuda.empty_cache()
 dump(ROOT/'status.json',{'state':'complete','seeds':args.seeds})
 print('COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[101,102,103]);p.add_argument('--probe-epochs',type=int,default=100)
 main(p.parse_args())
