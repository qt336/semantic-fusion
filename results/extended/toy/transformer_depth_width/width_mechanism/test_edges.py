"""Paired, frozen forward passes on the same 7,680 seed-106 inputs."""
from pathlib import Path
import json,sys
import numpy as np
import torch
from model import ToyModel

def run():
    wide,narrow,data,out=map(Path,sys.argv[1:])
    conds=['baseline','ordinary_a','block_b_from_a_first','block_displayed_q_from_b','block_displayed_q_from_a','block_final_query_from_b','block_final_query_from_a']
    base_modes=conds.copy()
    conds += ['ordinary_a+'+c for c in base_modes if c.startswith('block')]
    with np.load(data) as d:
        x,pos,edge,answer,first=[np.concatenate([d[f'semantic_{i}_{k}'] for i in range(3)]) for k in ('x','pos','edge','answer','first')]
    rows=np.arange(len(x)); bp,ap=edge.T; qp=bp+1
    assert np.all(x[rows,qp]==x[rows,pos]) and np.all(x[rows,bp]==answer)
    rng=np.random.default_rng(20261125)
    ordinary=np.array([rng.choice(np.setdiff1d(np.arange(3,331),row)) for row in x])
    # Only expose the prefix up to the queried position, including in the substitution condition.
    x[np.arange(x.shape[1])[None,:]>pos[:,None]]=335
    input0=torch.from_numpy(x).cuda()
    read,after,b,a,r,alt=[torch.as_tensor(z,device='cuda') for z in (pos,qp,bp,ap,rows,ordinary)]
    metrics={}; raw={}
    for width,path in [(1024,wide),(256,narrow)]:
        ck=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        epoch=ck['epoch']
        model=ToyModel('transformer',width=width,heads=8,layers=8).cuda().eval()
        model.load_state_dict(ck['model']);del ck
        metrics[width]={'epoch':epoch,'conditions':{}}
        for mode in conds:
            edge_mode=mode.split('+')[-1]
            preds=[]
            for start in range(0,len(x),128):
                sl=slice(start,start+128); ids=input0[sl].clone()
                ix=torch.arange(len(ids),device='cuda')
                if mode.startswith('ordinary_a'): ids[ix,a[sl]]=alt[sl]
                h=model.embedding(ids)
                for i,layer in enumerate(model.blocks):
                    blocked=None
                    if edge_mode=='block_b_from_a_first' and i==0: blocked=torch.stack([b[sl],a[sl]],1)
                    if edge_mode.startswith(('block_displayed','block_final')):
                        q=after if 'displayed' in edge_mode else read
                        k=b if edge_mode.endswith('_b') else a
                        blocked=torch.stack([q[sl],k[sl]],1)
                    h=layer(h,blocked)
                logits=model.head(model.norm(h[ix,read[sl]]))
                preds.append(logits.argmax(1).cpu().numpy())
            pred=np.concatenate(preds); raw[f'w{width}_{mode}']=pred
            m={'correct':int((pred==answer).sum()),'first':int((pred==(ordinary if mode.startswith('ordinary_a') else first)).sum()),'n':len(pred),'correct_by_pair':[int(t.sum()) for t in np.split(pred==answer,3)]}
            metrics[width]['conditions'][mode]=m
            print(width,epoch,mode,m,flush=True)
        del model
    np.savez_compressed(out/'predictions.npz',gold=answer,first=first,ordinary_a=ordinary,**raw)
    (out/'edge_results.json').write_text(json.dumps(metrics,indent=2)+'\n')
if __name__=='__main__':
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    with torch.inference_mode(): run()
