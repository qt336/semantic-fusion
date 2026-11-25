"""Rank sweep of independently identified constituent and arithmetic directions."""
from pathlib import Path
import argparse,json,hashlib,time,sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'subspace_intervention'
sys.path.insert(0,str(PREVIOUS))
from model import ToyModel

RANKS=(1,4,16,64,128,256)
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def run(width, first_layer=1, last_layer=8):
    torch.set_num_threads(3);torch.set_float32_matmul_precision('high')
    dest=ROOT/f'w{width}';dest.mkdir(exist_ok=True)
    saved=torch.load(PREVIOUS/f'w{width}/backbone.pt',map_location='cpu',mmap=True,weights_only=True)
    epoch=saved['epoch']
    model=ToyModel('transformer',width=width,heads=8,layers=8).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model']);del saved
    source=Path('./source_L8_D1024/datasets/seed_106.npz')
    data=np.load(source)
    tests={}
    for key in ['arithmetic_0','arithmetic_1','arithmetic_2','semantic_0','semantic_1','semantic_2','standard']:
        x=data[key+'_x'].copy();pos=data[key+'_pos'];gold=data[key+'_answer']
        bp=data[key+'_edge'][:,0].copy()
        if key=='standard':
            for j,(xs,p) in enumerate(zip(x,pos)):
                sep=int(np.flatnonzero(xs==332)[0]);bp[j]=int(np.flatnonzero(xs[:sep]==xs[p])[0])-1
                assert xs[bp[j]]==gold[j]
        x[np.arange(x.shape[1])[None,:]>pos[:,None]]=335
        tests[key]=[torch.tensor(z,device='cuda') for z in (x,bp,pos,gold,data[key+'_first'])]
    with torch.no_grad():
        baseline={}
        for key,(x,bp,pos,gold,first) in tests.items():
            output=[]
            for start in range(0,len(x),256):
                h=model.embedding(x[start:start+256])
                for block in model.blocks:
                    att=block.mixer(block.ln1(h));mlp=block.mlp(block.ln2(h+att));h=h+(att+mlp)
                ix=torch.arange(len(h),device='cuda')
                output.append(model.head(model.norm(h[ix,pos[start:start+256]])).argmax(1))
            baseline[key]=torch.cat(output)
    np.savez_compressed(dest/'baseline_predictions.npz',**{k:v.cpu().numpy() for k,v in baseline.items()})
    def scores(preds):
        result={}
        for key,pred in preds.items():
            _,_,_,gold,first=tests[key]
            correct=pred==gold
            counts={'correct':int(correct.sum()),'first':int(((pred==first)&~correct).sum()),'eos':int((pred==334).sum())}
            counts['other']=len(pred)-sum(counts.values())
            result[key]={'n':len(pred),'counts':counts}
        for group in ['arithmetic','semantic']:
            parts=[v for k,v in result.items() if k.startswith(group+'_')]
            result[group]={'n':sum(v['n'] for v in parts),'counts':{k:sum(v['counts'][k] for v in parts) for k in ['correct','first','eos','other']}}
        return result
    write(dest/'baseline.json',{'width':width,'epoch':epoch,'metrics':scores(baseline)})
    # Reproduce training-only arithmetic gradients; estimate them independently of the probe.
    basis_file=dest/'bases.pt'
    if not basis_file.exists():
        rng=np.random.default_rng(20261125)
        selected=np.concatenate([rng.choice(90000,256,replace=False)+90000*i for i in range(3)])
        ax=data['train_x'][selected,:5];gold=data['train_y'][selected,4]
        gradients=[[] for _ in range(8)]
        for start in range(0,len(ax),128):
            x=torch.tensor(ax[start:start+128],device='cuda');g=torch.tensor(gold[start:start+128],device='cuda')
            h=model.embedding(x).detach().requires_grad_(True);states=[]
            for block in model.blocks:
                h=block(h);h.retain_grad();states.append(h)
            logits=model.head(model.norm(h[:,4]))
            (logits.gather(1,g[:,None]).squeeze()-logits[:,:331].mean(1)).sum().backward()
            for i,h in enumerate(states):gradients[i].append(h.grad[:,3].detach().cpu())
        bases={}
        for layer in range(8):
            reader=torch.load(PREVIOUS/f'w{width}/layer{layer+1:02}_directions.pt',weights_only=True)['reader']
            weights=(reader['weight']-reader['weight'].mean(0))/reader['sd'][None,:]
            _,sb,vb=torch.linalg.svd(weights.double(),full_matrices=False)
            gradient=torch.cat(gradients[layer]).double()
            _,sa,va=torch.linalg.svd(gradient,full_matrices=False)
            gen=torch.Generator().manual_seed(4100+layer*10)
            random0=torch.linalg.qr(torch.randn(width,256,dtype=torch.float64,generator=gen),mode='reduced').Q
            gen=torch.Generator().manual_seed(4101+layer*10)
            random1=torch.linalg.qr(torch.randn(width,256,dtype=torch.float64,generator=gen),mode='reduced').Q
            bases[layer]={'b':vb[:256].T,'b_singular':sb,'arithmetic':va[:256].T,'arithmetic_singular':sa,'random0':random0,'random1':random1}
        torch.save(bases,basis_file)
    bases=torch.load(basis_file,weights_only=True)
    geometry=[]
    for layer in range(8):
        b=bases[layer]
        for rank in RANKS:
            overlap=float(((b['b'][:,:rank].T@b['arithmetic'][:,:rank])**2).sum()/rank) if b['arithmetic_singular'][0]>1e-9 else None
            geometry.append({'layer':layer+1,'rank':rank,'fraction':rank/width,'mean_squared_cosine':overlap,
                'b_spectrum_energy':float(b['b_singular'][:rank].square().sum()/b['b_singular'].square().sum())})
    write(dest/'geometry.json',geometry)
    completed=0
    for layer in range(8):
        if not first_layer<=layer+1<=last_layer:continue
        # Cache the unmodified state at this layer, then recompute the downstream blocks.
        cache={}
        with torch.no_grad():
            for key,(x,bp,pos,gold,first) in tests.items():
                hs=[];us=[]
                for start in range(0,len(x),256):
                    h=model.embedding(x[start:start+256])
                    for block in model.blocks[:layer+1]:
                        att=block.mixer(block.ln1(h));u=att+block.mlp(block.ln2(h+att));h=h+u
                    ix=torch.arange(len(h),device='cuda')
                    hs.append(h);us.append(u[ix,bp[start:start+256]])
                cache[key]=(torch.cat(hs),torch.cat(us))
            # Check that replaying the cached suffix reproduces the same baseline exactly.
            for key,(h,u) in cache.items():
                bp,pos=tests[key][1:3];output=[]
                for start in range(0,len(h),256):
                    state=h[start:start+256]
                    for block in model.blocks[layer+1:]:
                        att=block.mixer(block.ln1(state));state=state+(att+block.mlp(block.ln2(state+att)))
                    ix=torch.arange(len(state),device='cuda')
                    output.append(model.head(model.norm(state[ix,pos[start:start+256]])).argmax(1))
                assert torch.equal(torch.cat(output),baseline[key]),(layer,key)
        for kind in ['b','arithmetic']:
            if kind=='arithmetic' and bases[layer]['arithmetic_singular'][0]<=1e-9:continue
            for rank in RANKS:
                basis=bases[layer][kind][:,:rank].float().cuda()
                assert float((basis.double().T@basis.double()-torch.eye(rank,device='cuda')).abs().max())<2e-5
                for variant in ['target','random0','random1']:
                    name=f'layer{layer+1:02}_{kind}_rank{rank:03}_{variant}'
                    path=dest/f'{name}.json'
                    if path.exists(): completed+=1;continue
                    random=bases[layer][variant][:,:rank].float().cuda() if variant!='target' else None
                    predictions={};perturbation={}
                    with torch.no_grad():
                        for key,(h,u) in cache.items():
                            bp,pos=tests[key][1:3]
                            delta=(u@basis)@basis.T
                            lengths=delta.norm(dim=-1)
                            if random is not None:
                                rand=(u@random)@random.T
                                delta=rand*(lengths/rand.norm(dim=-1).clamp_min(1e-12))[:,None]
                                torch.testing.assert_close(delta.norm(dim=-1),lengths,rtol=3e-5,atol=1e-5)
                            perturbation[key]={'mean_norm':float(lengths.mean()),'mean_update_norm':float(u.norm(dim=-1).mean())}
                            output=[]
                            for start in range(0,len(h),256):
                                state=h[start:start+256].clone();ix=torch.arange(len(state),device='cuda')
                                state[ix,bp[start:start+256]]-=delta[start:start+256]
                                for block in model.blocks[layer+1:]:
                                    att=block.mixer(block.ln1(state));state=state+(att+block.mlp(block.ln2(state+att)))
                                output.append(model.head(model.norm(state[ix,pos[start:start+256]])).argmax(1))
                            predictions[key]=torch.cat(output)
                            if layer==7:assert torch.equal(predictions[key],baseline[key])
                    m=scores(predictions)
                    np.savez_compressed(dest/f'{name}.npz',**{k:v.cpu().numpy() for k,v in predictions.items()})
                    write(path,{'width':width,'epoch':epoch,'layer':layer+1,'kind':kind,'rank':rank,'variant':variant,'metrics':m,'perturbation':perturbation})
                    completed+=1
                    write(dest/f'progress_{first_layer}_{last_layer}.json',{'width':width,'completed':completed,'total':270,'layer':layer+1,'kind':kind,'rank':rank})
                    print(width,layer+1,kind,rank,variant,[round(100*m[k]['counts']['correct']/m[k]['n'],2) for k in ['arithmetic','standard','semantic']],flush=True)
        del cache
    write(dest/f'worker_{first_layer}_{last_layer}.json',{'width':width,'epoch':epoch,'conditions':completed,'cached_suffix_verified':True,'rank_grid':list(RANKS)})
    if len(list(dest.glob('layer*.json')))==270:
        write(dest/'complete.json',{'width':width,'epoch':epoch,'conditions':270,'cached_suffix_verified':True,'rank_grid':list(RANKS)})
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('width',type=int,choices=[256,1024]);parser.add_argument('--first',type=int,default=1);parser.add_argument('--last',type=int,default=8);args=parser.parse_args()
    run(args.width,args.first,args.last)
