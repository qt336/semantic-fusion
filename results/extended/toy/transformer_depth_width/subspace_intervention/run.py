"""Frozen forward interventions with ordinary-token probes and arithmetic gradients."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from model import ToyModel
from qwen_fit import fit

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]/'toy_transformer_8layer_d1024_20261125'
# Resolve shared inputs from the original experiment folder.
BASE=Path('./source_L8_D1024')
NARROW=Path('./source_L8_D256')
RANK=16

def atomic(path,data):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(path)

def features(model,x,bp):
    all_features=torch.empty(8,len(x),model.embedding.embedding_dim)
    for start in range(0,len(x),256):
        ids=torch.tensor(x[start:start+256],device='cuda')
        b=torch.tensor(bp[start:start+256],device='cuda')
        h=model.embedding(ids)
        row=torch.arange(len(ids),device='cuda')
        for i,block in enumerate(model.blocks):
            att=block.mixer(block.ln1(h))
            mlp=block.mlp(block.ln2(h+att))
            update=att+mlp
            all_features[i,start:start+len(ids)]=update[row,b].detach().cpu()
            h=h+update
    return all_features

def calc(model,x,bp,pos,layer=-1,basis=None,random_basis=None):
    h=model.embedding(x);row=torch.arange(len(x),device=x.device)
    norms=[]
    for i,block in enumerate(model.blocks):
        att=block.mixer(block.ln1(h))
        mlp=block.mlp(block.ln2(h+att));u=att+mlp
        h=h+u
        if i==layer:
            delta=(u[row,bp]@basis)@basis.T
            norms.append(delta.norm(dim=-1))
            if random_basis is not None:
                rand=(u[row,bp]@random_basis)@random_basis.T
                delta=rand*(delta.norm(dim=-1)/rand.norm(dim=-1).clamp_min(1e-12))[:,None]
            h=h.clone();h[row,bp]-=delta
    return model.head(model.norm(h[row,pos])),norms

def run(width):
    dest=ROOT/f'w{width}';dest.mkdir(exist_ok=True)
    path=BASE/'runs/transformer_seed106/latest.pt' if width==1024 else NARROW/'runs/transformer_seed106/latest.pt'
    frozen=dest/'backbone.pt'
    if not frozen.exists():
        ck=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        torch.save({'model':ck['model'],'epoch':ck['epoch'],'step':ck['step'],'width':width},frozen)
    ck=torch.load(frozen,map_location='cpu',weights_only=True,mmap=True)
    epoch=ck['epoch'];model=ToyModel('transformer',width=width,heads=8,layers=8).cuda().eval().requires_grad_(False)
    model.load_state_dict(ck['model']);del ck
    ordinary=np.load(BASE/'all_layers_probe_seed106/ordinary.npz')
    data=np.load(BASE/'datasets/seed_106.npz')
    cache={}
    with torch.no_grad():
        for part in ['train','val','test']:
            x=ordinary[part+'_x'].copy();bp=ordinary[part+'_bpos']
            x[np.arange(x.shape[1])[None,:]>bp[:,None]]=335
            cache[part]=features(model,x,bp)
    tests={}
    for key in ['arithmetic_0','arithmetic_1','arithmetic_2','semantic_0','semantic_1','semantic_2','standard']:
        x=data[key+'_x'].copy();pos=data[key+'_pos'];gold=data[key+'_answer']
        bp=data[key+'_edge'][:,0].copy()
        if key=='standard':
            for j,(xs,p) in enumerate(zip(x,pos)):
                sep=int(np.flatnonzero(xs==332)[0]);q=xs[p]
                bp[j]=int(np.flatnonzero(xs[:sep]==q)[0])-1
                assert xs[bp[j]]==gold[j]
        x[np.arange(x.shape[1])[None,:]>pos[:,None]]=335
        tests[key]=(x,bp,pos,gold,data[key+'_first'])
    def evaluate(layer=-1,basis=None,random_basis=None):
        metrics={};preds={}
        with torch.no_grad():
            for key,(x,bp,pos,gold,first) in tests.items():
                output=[];energies=[]
                for start in range(0,len(x),256):
                    ids,b,p=[torch.tensor(z[start:start+256],device='cuda') for z in (x,bp,pos)]
                    logits,norm=calc(model,ids,b,p,layer,basis,random_basis)
                    output.append(logits.argmax(-1).cpu().numpy())
                    if norm: energies.extend(norm[0].cpu().tolist())
                pred=np.concatenate(output);preds[key]=pred
                metrics[key]={'n':len(pred),'correct':int((pred==gold).sum()),'first':int(((pred==first)&(pred!=gold)).sum()),'eos':int((pred==334).sum()),'mean_removed_norm':float(np.mean(energies)) if energies else 0}
        for group in ['arithmetic','semantic']:
            chosen=[v for k,v in metrics.items() if k.startswith(group+'_')]
            metrics[group]={k:sum(v[k] for v in chosen) for k in ['n','correct','first','eos']}
        return metrics,preds
    metrics,preds=evaluate()
    atomic(dest/'baseline.json',{'epoch':epoch,'metrics':metrics})
    np.savez_compressed(dest/'baseline_predictions.npz',**preds)
    # Arithmetic direction discovery uses training examples only, balanced over operations.
    rng=np.random.default_rng(20261125)
    ids=np.concatenate([rng.choice(90000,256,replace=False)+90000*i for i in range(3)])
    ax=data['train_x'][ids,:5];gold=data['train_y'][ids,4]
    gradients=[[] for _ in range(8)]
    for start in range(0,len(ax),128):
        x=torch.tensor(ax[start:start+128],device='cuda');g=torch.tensor(gold[start:start+128],device='cuda')
        h=model.embedding(x).detach().requires_grad_(True);states=[]
        for block in model.blocks:
            h=block(h);h.retain_grad();states.append(h)
        logits=model.head(model.norm(h[:,4]))
        margin=logits.gather(1,g[:,None]).squeeze()-logits[:,:331].mean(1)
        margin.sum().backward()
        for i,h in enumerate(states): gradients[i].append(h.grad[:,3].detach().cpu())
    for layer in range(8):
        f=dest/f'layer{layer+1:02}.json'
        if f.exists() and json.loads(f.read_text()).get('complete') and json.loads(f.read_text()).get('double_orthogonal'):continue
        saved_path=dest/f'layer{layer+1:02}_directions.pt'
        if saved_path.exists():
            reader=torch.load(saved_path,weights_only=True)['reader']
        else:
            reader=fit(cache['train'][layer],torch.tensor(ordinary['train_labels'][:,1]),cache['val'][layer],torch.tensor(ordinary['val_labels'][:,1]),'cuda')
        # Convert the standardized classifier into raw-coordinate logit contrasts.
        weight=(reader['weight']-reader['weight'].mean(0))/reader['sd'][None,:]
        _,_,vh=torch.linalg.svd(weight.double(),full_matrices=False)
        b_cpu=vh[:RANK].T
        b_basis=b_cpu.float().cuda()
        grads=torch.cat(gradients[layer]).double()
        # Search for arithmetic-sensitive directions orthogonal to the b directions.
        grads=grads-(grads@b_cpu)@b_cpu.T
        _,singular,vh=torch.linalg.svd(grads,full_matrices=False)
        a_cpu=vh[:RANK].T
        a_cpu=a_cpu-b_cpu@(b_cpu.T@a_cpu)
        a_basis=torch.linalg.qr(a_cpu,mode='reduced').Q.float().cuda()
        conditions={'b':b_basis}
        # At the final block, b cannot influence a later answer position any more.
        if float(singular[0])>1e-9: conditions['arithmetic']=a_basis
        rec={'epoch':epoch,'width':width,'layer':layer+1,'rank':RANK,'double_orthogonal':True,'arithmetic_gradient_norm':float(grads.norm()),'probe_epoch':reader['epoch'],'probe_test':{},'conditions':{}}
        with torch.no_grad():
            z=(cache['test'][layer]-reader['mean'])/reader['sd']
            pred=F.linear(z,reader['weight'],reader['bias']).argmax(1).numpy()
            rec['probe_test']={'correct':int((pred==ordinary['test_labels'][:,1]).sum()),'n':len(pred)}
            for group in ['semantic_0','semantic_1','semantic_2']:
                x,bp,_,g,_=tests[group]
                feats=features(model,x,bp)[layer]
                pred=F.linear((feats-reader['mean'])/reader['sd'],reader['weight'],reader['bias']).argmax(1).numpy()
                rec['probe_test'][group]={'correct':int((pred==g).sum()),'n':len(pred)}
        saved={'reader':reader,'b_basis':b_basis.cpu()}
        if 'arithmetic' in conditions: saved['arithmetic_basis']=a_basis.cpu()
        torch.save(saved,dest/f'layer{layer+1:02}_directions.pt')
        for name,basis in conditions.items():
            for repeat in [-1,0,1]:
                random_basis=None
                if repeat>=0:
                    gen=torch.Generator().manual_seed(4100+layer*10+repeat)
                    random_basis=torch.linalg.qr(torch.randn(width,RANK,generator=gen,dtype=torch.float64),mode='reduced').Q.float().cuda()
                label=name if repeat<0 else f'{name}_random{repeat}'
                m,p=evaluate(layer,basis,random_basis)
                rec['conditions'][label]=m
                np.savez_compressed(dest/f'layer{layer+1:02}_{label}_predictions.npz',**p)
                atomic(f,rec)
                print(width,epoch,layer+1,label,[round(m[k]['correct']/m[k]['n']*100,2) for k in ['arithmetic','standard','semantic']],flush=True)
        rec['complete']=True;atomic(f,rec)
    atomic(dest/'complete.json',{'width':width,'epoch':epoch})
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('width',type=int);args=parser.parse_args()
    torch.set_num_threads(3);torch.set_float32_matmul_precision('high')
    run(args.width)
