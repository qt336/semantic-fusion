"""Frozen native-update probes; ordinary validation only, no backbone training."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import ToyModel
from qwen_fit import fit

ROOT=Path(__file__).resolve().parent
def save(path, obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

@torch.no_grad()
def features(model,x,bp):
    result=torch.empty(8,len(x),model.embedding.embedding_dim)
    for start in range(0,len(x),128):
        ids=torch.as_tensor(x[start:start+128],device='cuda')
        b=torch.as_tensor(bp[start:start+128],device='cuda')
        h=model.embedding(ids);ii=torch.arange(len(ids),device='cuda')
        for layer,block in enumerate(model.blocks):
            att=block.mixer(block.ln1(h));mlp=block.mlp(block.ln2(h+att));u=att+mlp
            result[layer,start:start+len(ids)]=u[ii,b].cpu()
            if start==0:
                torch.testing.assert_close(h+u,block(h),rtol=2e-4,atol=2e-5)
            h=h+u
    return result

def main(seed,width,allow_underqualified=False):
    torch.set_num_threads(3);torch.set_float32_matmul_precision('high');torch.manual_seed(20261125)
    dest=ROOT/f'seed{seed}_w{width}';dest.mkdir(exist_ok=True)
    checkpoint=ROOT/f'inputs/seed{seed}_w{width}.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model=ToyModel('transformer',width=width,heads=8,layers=8).cuda().eval().requires_grad_(False)
    model.load_state_dict(ck['model']);epoch=ck['epoch'];del ck
    digest=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    metadata=dict(seed=seed,width=width,epoch=epoch,checkpoint_sha256=digest.hexdigest(),
                  representation='attention output + normal MLP output at displayed b; block input residual excluded',
                  ordinary_train=21184,ordinary_validation=2648,probe_epochs=100)
    save(dest/'metadata.json',metadata)
    ordinary=np.load(ROOT/'inputs/ordinary.npz');data=np.load(ROOT/f'inputs/seed_{seed}.npz')
    metrics={};predictions={}
    with torch.no_grad():
        for key in ['arithmetic_0','arithmetic_1','arithmetic_2','standard','semantic_0','semantic_1','semantic_2']:
            x=data[key+'_x'].copy();pos=data[key+'_pos'];gold=data[key+'_answer']
            x[np.arange(x.shape[1])[None,:]>pos[:,None]]=335
            preds=[]
            for start in range(0,len(x),128):
                ids=torch.as_tensor(x[start:start+128],device='cuda');p=torch.as_tensor(pos[start:start+128],device='cuda')
                logits=model(ids)[torch.arange(len(ids),device='cuda'),p]
                preds.append(logits.argmax(-1).cpu().numpy())
            pred=np.concatenate(preds);predictions[key]=pred
            metrics[key]=dict(correct=int((pred==gold).sum()),n=len(gold))
        for name in ['arithmetic','semantic']:
            ms=[v for k,v in metrics.items() if k.startswith(name+'_')]
            metrics[name]={k:sum(v[k] for v in ms) for k in ['correct','n']}
    save(dest/'baseline.json',metrics);np.savez_compressed(dest/'baseline_predictions.npz',**predictions)
    qualified=all(metrics[k]['correct']/metrics[k]['n']>=.99 for k in ['arithmetic','standard'])
    metadata['qualified']=qualified;save(dest/'metadata.json',metadata)
    assert qualified or allow_underqualified,metrics
    cache={};labels={}
    for part in ['train','val','test']:
        x=ordinary[part+'_x'].copy();bp=ordinary[part+'_bpos'];labels[part]=ordinary[part+'_labels'][:,1]
        assert np.array_equal(x[np.arange(len(x)),bp],labels[part])
        x[np.arange(x.shape[1])[None,:]>bp[:,None]]=335
        cache[part]=features(model,x,bp)
        print(seed,width,'features',part,flush=True)
    for key in ['semantic_0','semantic_1','semantic_2']:
        x=data[key+'_x'].copy();bp=data[key+'_edge'][:,0];labels[key]=data[key+'_answer']
        assert np.array_equal(x[np.arange(len(x)),bp],labels[key])
        x[np.arange(x.shape[1])[None,:]>bp[:,None]]=335
        cache[key]=features(model,x,bp)
    for layer in range(8):
        if (dest/f'layer{layer+1:02}.json').exists():continue
        start=time.time()
        reader=fit(cache['train'][layer],torch.as_tensor(labels['train']),cache['val'][layer],torch.as_tensor(labels['val']),'cuda')
        rec=dict(layer=layer+1,probe_epoch=reader['epoch']);preds={}
        with torch.no_grad():
            for name in ['test','semantic_0','semantic_1','semantic_2']:
                z=(cache[name][layer]-reader['mean'])/reader['sd']
                pred=F.linear(z,reader['weight'],reader['bias']).argmax(1).numpy()
                preds[name]=pred;preds[name+'_gold']=labels[name]
                rec[name]=dict(correct=int((pred==labels[name]).sum()),n=len(pred))
        rec['semantic']={k:sum(rec[f'semantic_{i}'][k] for i in range(3)) for k in ['correct','n']}
        torch.save(reader,dest/f'layer{layer+1:02}_reader.pt')
        np.savez_compressed(dest/f'layer{layer+1:02}_predictions.npz',**preds)
        save(dest/f'layer{layer+1:02}.json',rec)
        print(seed,width,layer+1,rec,'seconds',round(time.time()-start,1),flush=True)
        subprocess.run([sys.executable,str(ROOT/'report.py')],check=True)
    save(dest/'complete.json',metadata)
    subprocess.run([sys.executable,str(ROOT/'report.py')],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('seed',type=int);p.add_argument('width',type=int);p.add_argument('--allow-underqualified',action='store_true');a=p.parse_args();main(a.seed,a.width,a.allow_underqualified)
