from pathlib import Path
import json,time,hashlib,math,os
import numpy as np
import torch
from torch.nn import functional as F
from model import ToyModel
from features import components
from probe_common import SOURCE,read_special,PAIRS,dump
ROOT=Path(__file__).parent
BASE=ROOT.parent/'probe_data.npz'
CONFIGS=[{'name':'base_zero','lr':.01,'wd':.01,'init':0},
 {'name':'base_init1','lr':.01,'wd':.01,'init':1},{'name':'base_init2','lr':.01,'wd':.01,'init':2},
 {'name':'lr003','lr':.003,'wd':.01,'init':0},{'name':'lr03','lr':.03,'wd':.01,'init':0},
 {'name':'wd0','lr':.01,'wd':0.,'init':0},{'name':'wd01','lr':.01,'wd':.1,'init':0},
 {'name':'lr003_wd0','lr':.003,'wd':0.,'init':0},{'name':'lr03_wd01','lr':.03,'wd':.1,'init':0}]

def log(*args):print(*args,flush=True)
def status(**kw):dump(ROOT/'status.json',{'updated_unix':time.time(),**kw})

def datasets():
 with np.load(BASE) as z:base={s:{k:z[f'{s}_{k}'] for k in ['x','labels','bpos','qpos']} for s in ['train','val','test']}
 seen=set()
 for d in base.values():
  for row,bp,qp in zip(d['x'],d['bpos'],d['qpos']):
   n=int(qp)-3;sig=(*row[1:n+1].tolist(),-1,int(bp));assert sig not in seen;seen.add(sig)
 rng=np.random.default_rng(20261108);out={}
 for name,reps in [('extra',192),('fresh_test',32)]:
  xx=[];ll=[];bb=[];qq=[]
  for _ in range(reps):
   while True:
    bs=rng.permutation(331)
    if all(a!=b and (a,int(b)) not in PAIRS for a,b in enumerate(bs)):break
   for a,b in enumerate(bs):
    b=int(b)
    while True:
     n=int(rng.integers(3,7));start=int(rng.integers(n-2));pool=[i for i in range(331) if i not in [a,b]]
     others=rng.choice(pool,n-2,replace=False).tolist();mem=others[:start]+[a,b]+others[start:];query=mem[start+2]
     sig=(*mem,-1,start+2)
     if sig in seen or any(pair in PAIRS for pair in zip(mem,mem[1:])) or (query,b) in PAIRS:continue
     seen.add(sig);break
    row=[333,*mem,332,331,query,b,334];xx.append(row+[335]*(12-len(row)));ll.append([a,b]);bb.append(start+2);qq.append(n+3)
  out[name]={'x':np.array(xx,dtype=np.int64)[:,:-1],'labels':np.array(ll),'bpos':np.array(bb),'qpos':np.array(qq)}
  for j in [0,1]:assert np.all(np.bincount(out[name]['labels'][:,j],minlength=331)==reps)
 base.update(out);base['large']={k:np.concatenate([base['train'][k],base['extra'][k]]) for k in base['train']}
 np.savez_compressed(ROOT/'datasets.npz',**{f'{s}_{k}':v for s,d in base.items() for k,v in d.items()})
 dump(ROOT/'data_audit.json',{'counts':{s:len(d['x']) for s,d in base.items()},'disjoint_train_val_test_extra_fresh':True,'balanced_labels':True,'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest()})
 return base

def matched(seed,data,special):
 used=set(map(tuple,data['large']['labels'].tolist()))|set(map(tuple,data['val']['labels'].tolist()))
 rng=np.random.default_rng(20261109);result={}
 for group in ['operator_0_1','operator_0_2','operator_2_1']:
  d=special[group];xx=[];ll=[];bb=[];qq=[];indices=[]
  for i,(row,bp,qp,label) in enumerate(zip(d['x'],d['bpos'],d['qpos'],d['labels'])):
   b=int(label[1]);end=int(np.flatnonzero(row==332)[0]);mem=row[1:end]
   options=[]
   for a in range(3,331):
    if a in mem or (a,b) in used or (a,b) in PAIRS:continue
    trial=row.copy();trial[bp-1]=a
    if any(pair in PAIRS for pair in zip(trial[1:end-1],trial[2:end])):continue
    options.append(a)
   assert len(options)>=8
   for a in rng.choice(options,8,replace=False):
    trial=row.copy();trial[bp-1]=a;xx.append(trial);ll.append([a,b]);bb.append(bp);qq.append(qp);indices.append(i)
  result[group.replace('operator_','matched8_')]={'x':np.array(xx),'labels':np.array(ll),'bpos':np.array(bb),'qpos':np.array(qq),'context_index':np.array(indices)}
 np.savez_compressed(ROOT/f'matched8_seed{seed}.npz',**{f'{s}_{k}':v for s,d in result.items() for k,v in d.items()})
 return result

@torch.no_grad()
def extract(model,d,masked=False):
 result=[]
 for start in range(0,len(d['x']),512):
  sl=slice(start,start+512);x=torch.as_tensor(d['x'][sl],device='cuda');bp=torch.as_tensor(d['bpos'][sl],device='cuda')
  f,_=components(model,x,bp,masked)
  result.append(f[:,5])
 return torch.cat(result)

@torch.no_grad()
def score(x,y,w,b):
 # Return one score per independent affine head, accumulating over batches.
 acc=torch.zeros(w.shape[0],device=x.device);nll=acc.clone()
 for start in range(0,len(x),2048):
  sl=slice(start,start+2048);logits=x[sl][None]@w+b[:,None,:]
  acc+=(logits.argmax(-1)==y[sl][None]).sum(1)
  losses=F.cross_entropy(logits.reshape(-1,331),y[sl][None].expand(len(w),-1).reshape(-1),reduction='none').reshape(len(w),-1)
  nll+=losses.sum(1)
 return acc/len(x),nll/len(x)

@torch.no_grad()
def predict(x,w,b):
 return torch.cat([(x[s:s+2048]@w+b).argmax(-1).cpu() for s in range(0,len(x),2048)]).numpy()

def adam_grid(x,y,val,vy,configs,epochs,tag,seed):
 weights=[];biases=[];groups=[]
 for c in configs:
  gen=torch.Generator(device=x.device).manual_seed(711+c['init'])
  init=torch.zeros((256,331),device=x.device) if c['init']==0 else .01*torch.randn((256,331),device=x.device,generator=gen)
  w=torch.nn.Parameter(init);b=torch.nn.Parameter(torch.zeros(331,device=x.device));weights.append(w);biases.append(b)
  groups.extend([{'params':[w],'lr':c['lr'],'weight_decay':c['wd']},{'params':[b],'lr':c['lr'],'weight_decay':0.}])
 opt=torch.optim.AdamW(groups);g=torch.Generator(device=x.device).manual_seed(812)
 bestacc=torch.full((len(configs),),-1.,device=x.device);bestloss=torch.full_like(bestacc,float('inf'))
 bestw=torch.stack(weights).detach().clone();bestb=torch.stack(biases).detach().clone();beste=torch.zeros(len(configs),dtype=torch.long,device=x.device)
 hist=[];snapshots={};started=time.monotonic()
 for epoch in range(1,epochs+1):
  order=torch.randperm(len(x),generator=g,device=x.device);loss_sum=torch.zeros(len(configs),device=x.device)
  for off in range(0,len(x),2048):
   ids=order[off:off+2048];w=torch.stack(weights);b=torch.stack(biases)
   logits=x[ids][None]@w+b[:,None,:]
   loss=F.cross_entropy(logits.reshape(-1,331),y[ids][None].expand(len(configs),-1).reshape(-1),reduction='none').reshape(len(configs),-1).mean(1)
   opt.zero_grad();loss.sum().backward();opt.step();loss_sum+=loss.detach()*len(ids)
  if epoch==1 or epoch%25==0 or epoch==epochs:
   with torch.no_grad():
    w=torch.stack(weights);b=torch.stack(biases);acc,nll=score(val,vy,w,b)
    update=(acc>bestacc)|((acc==bestacc)&(nll<bestloss));bestacc[update]=acc[update];bestloss[update]=nll[update];bestw[update]=w[update];bestb[update]=b[update];beste[update]=epoch
    hist.append({'epoch':epoch,'train_nll':(loss_sum/len(x)).cpu().tolist(),'val_acc':acc.cpu().tolist(),'val_nll':nll.cpu().tolist(),'best_val_acc':bestacc.cpu().tolist(),'best_epoch':beste.cpu().tolist(),'seconds':time.monotonic()-started})
   dump(ROOT/f'{tag}_seed{seed}_history.json',hist)
   status(state='training_probes',model_seed=seed,stage=tag,epoch=epoch,target_epochs=epochs,configs=len(configs))
   if epoch%100==0:log(seed,tag,'epoch',epoch,'best val',bestacc.cpu().tolist())
  if epoch in [100,500,1000,2000] or epoch==epochs:
   snapshots[epoch]={'w':bestw.detach().clone(),'b':bestb.detach().clone(),'val_acc':bestacc.cpu().tolist(),'val_nll':bestloss.cpu().tolist(),'selected_epoch':beste.cpu().tolist()}
   torch.save({k:{kk:vv.cpu() if torch.is_tensor(vv) else vv for kk,vv in v.items()} for k,v in snapshots.items()},ROOT/f'{tag}_seed{seed}_snapshots.pt')
 return snapshots,hist

def ridge(x,y,v,vy):
 xd=torch.cat([x.double(),torch.ones((len(x),1),device=x.device,dtype=torch.float64)],1)
 vd=torch.cat([v.double(),torch.ones((len(v),1),device=x.device,dtype=torch.float64)],1)
 gram=xd.T@xd/len(x);rhs=xd.T@F.one_hot(y,331).double()/len(x);eye=torch.eye(257,device=x.device,dtype=torch.float64);eye[-1,-1]=0
 models=[]
 for alpha in [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.]:
  w=torch.linalg.solve(gram+alpha*eye,rhs);s=vd@w;acc=(s.argmax(-1)==vy).double().mean().item()
  models.append({'alpha':alpha,'w':w[:-1].float(),'b':w[-1].float(),'val_acc':acc})
 return models

def lbfgs(x,y,val,vy,reg,seed):
 xd=x.double();vd=val.double()
 w=torch.nn.Parameter(torch.zeros(256,331,device=x.device,dtype=torch.float64));b=torch.nn.Parameter(torch.zeros(331,device=x.device,dtype=torch.float64))
 opt=torch.optim.LBFGS([w,b],lr=1.,max_iter=100,max_eval=150,history_size=30,tolerance_grad=1e-6,tolerance_change=1e-12,line_search_fn='strong_wolfe')
 calls=0;trace=[]
 def closure():
  nonlocal calls
  opt.zero_grad();loss=F.cross_entropy(xd@w+b,y)+.5*reg*w.square().sum();loss.backward();calls+=1
  return loss
 converged=False
 for chunk in range(10):
  opt.step(closure);loss=float(closure().detach());grad=max(w.grad.abs().max().item(),b.grad.abs().max().item())
  trace.append({'chunk':chunk+1,'objective':loss,'gradient_inf':grad,'closure_calls':calls,'iterations':opt.state[w]['n_iter']})
  log(seed,'LBFGS',reg,'chunk',chunk+1,'gradient',grad)
  status(state='training_probes',model_seed=seed,stage='lbfgs',regularization=reg,chunk=chunk+1,gradient_inf=grad)
  if grad<=1e-6:converged=True;break
 with torch.no_grad():
  logits=vd@w+b;acc=(logits.argmax(-1)==vy).double().mean().item();nll=F.cross_entropy(logits,vy).item()
 return {'w':w.detach().float(),'b':b.detach().float(),'lambda':reg,'val_acc':acc,'val_nll':nll,'converged':converged,'trace':trace}

def evaluate_reader(w,b,features,groups,mean,sd):
 result={};preds={}
 for key,x in features.items():
  cond,group=key.split('/',1);prediction=predict((x-mean)/sd,w,b);preds[key]=prediction
  d=groups[group];gold=d['labels'][:,1];first=d['labels'][:,0]
  result[key]={'n':len(gold),'correct':int((prediction==gold).sum()),'accuracy':float((prediction==gold).mean()),'reads_A':float((prediction==first).mean())}
 return result,preds

def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(711)
 dump(ROOT/'manifest.json',{'configs':CONFIGS,'epochs':2000,'model_seeds':[101,102,103],'primary':'attention_mlp_no_embedding','target':'B','selection':'ordinary validation accuracy, then NLL','lbfgs_lambdas':[1e-6,1e-4,1e-2]})
 data=datasets();log('DATA READY')
 for seed in [101,102,103]:
  final=ROOT/f'results_seed{seed}.json'
  if final.exists():log('ALREADY COMPLETE',seed);continue
  status(state='extracting',model_seed=seed)
  ckpt=SOURCE/'runs'/f'heldout_pairs_seed{seed}'/'latest.pt';saved=torch.load(ckpt,map_location='cuda',weights_only=False);assert saved['epoch']==1200
  model=ToyModel().cuda().eval();model.load_state_dict(saved['model']);model.requires_grad_(False)
  x=torch.as_tensor(data['test']['x'][:32],device='cuda');bp=torch.as_tensor(data['test']['bpos'][:32],device='cuda')
  f,_=components(model,x,bp,False,audit=True)
  changed=x.clone()
  for i in range(len(x)):changed[i,bp[i]+1:]=17
  cf,_=components(model,changed,bp,False,audit=True);assert torch.equal(f,cf)
  raw_train=extract(model,data['train']);raw_val=extract(model,data['val']);raw_large=extract(model,data['large'])
  mean=raw_train.mean(0);sd=raw_train.std(0).clamp_min(1e-6);train=(raw_train-mean)/sd;val=(raw_val-mean)/sd
  y=torch.as_tensor(data['train']['labels'][:,1],device='cuda');vy=torch.as_tensor(data['val']['labels'][:,1],device='cuda')
  snapshots,hist=adam_grid(train,y,val,vy,CONFIGS,2000,'grid',seed)
  finalstate=snapshots[2000];best_index=max(range(len(CONFIGS)),key=lambda i:(finalstate['val_acc'][i],-finalstate['val_nll'][i]))
  best_config=CONFIGS[best_index];log('VALIDATION SELECTED',seed,best_config)
  ridge_models=ridge(train,y,val,vy)
  lbmodels=[lbfgs(train,y,val,vy,reg,seed) for reg in [1e-6,1e-4,1e-2]]
  lm=raw_large.mean(0);ls=raw_large.std(0).clamp_min(1e-6);large=(raw_large-lm)/ls;lv=(raw_val-lm)/ls
  ly=torch.as_tensor(data['large']['labels'][:,1],device='cuda')
  large_states,large_hist=adam_grid(large,ly,lv,vy,[best_config],2000,'large',seed)
  # Shuffled-label negative control, never used in model selection.
  gen=torch.Generator(device='cuda').manual_seed(20261110);shuffled_y=y[torch.randperm(len(y),device='cuda',generator=gen)]
  shuffled_states,_=adam_grid(train,shuffled_y,val,vy,[CONFIGS[0]],100,'shuffled',seed)
  specials=read_special(seed);matched_groups=matched(seed,data,specials)
  groups={'ordinary_test':data['test'],'ordinary_fresh_test':data['fresh_test'],**specials,**matched_groups}
  features={f'{cond}/{g}':extract(model,d,masked=cond=='masked_layer0') for cond in ['baseline','masked_layer0'] for g,d in groups.items()}
  readers=[]
  for budget,s in snapshots.items():
   for i,c in enumerate(CONFIGS):readers.append({'name':f'adam_{c["name"]}_budget{budget}','family':'adamw','budget':budget,'config':c,'w':s['w'][i],'b':s['b'][i],'mean':mean,'sd':sd,'val_acc':s['val_acc'][i],'val_nll':s['val_nll'][i],'epoch':s['selected_epoch'][i]})
  for rm in ridge_models:readers.append({'name':f'ridge_{rm["alpha"]}','family':'ridge','mean':mean,'sd':sd,**rm})
  for lmdo in lbmodels:readers.append({'name':f'lbfgs_{lmdo["lambda"]}','family':'lbfgs','mean':mean,'sd':sd,**lmdo})
  for budget,s in large_states.items():readers.append({'name':f'large_budget{budget}','family':'adamw_large','config':best_config,'budget':budget,'w':s['w'][0],'b':s['b'][0],'mean':lm,'sd':ls,'val_acc':s['val_acc'][0],'val_nll':s['val_nll'][0],'epoch':s['selected_epoch'][0]})
  s=shuffled_states[100];readers.append({'name':'shuffled','family':'negative_control','w':s['w'][0],'b':s['b'][0],'mean':mean,'sd':sd,'val_acc':s['val_acc'][0],'val_nll':s['val_nll'][0]})
  selected={'adamw':f'adam_{best_config["name"]}_budget2000','adamw_large':'large_budget2000','ridge':max((z for z in readers if z['family']=='ridge'),key=lambda z:z['val_acc'])['name'],'lbfgs':max((z for z in readers if z['family']=='lbfgs'),key=lambda z:(z['val_acc'],-z['val_nll']))['name']}
  output={'seed':seed,'selected':selected,'readers':[],'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'feature_checks_passed':True}
  pred_dir=ROOT/f'predictions_seed{seed}';pred_dir.mkdir(exist_ok=True)
  weights_save={}
  for reader in readers:
   ev,pred=evaluate_reader(reader['w'],reader['b'],features,groups,reader['mean'],reader['sd'])
   row={k:v for k,v in reader.items() if k not in ['w','b','mean','sd']};row['results']=ev;output['readers'].append(row)
   np.savez_compressed(pred_dir/(reader['name']+'.npz'),**pred)
   weights_save[reader['name']]={k:reader[k].cpu() for k in ['w','b','mean','sd']}
  torch.save(weights_save,ROOT/f'weights_seed{seed}.pt');dump(final,output)
  log('SEED COMPLETE',seed,selected)
  del model,raw_train,raw_val,raw_large,train,val,large,lv,features,readers,snapshots,large_states,shuffled_states,weights_save;torch.cuda.empty_cache()
 status(state='complete',seeds=[101,102,103]);log('ALL COMPLETE')

if __name__=='__main__':main()
