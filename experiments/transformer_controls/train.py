"""A single plain-Transformer run; random targets are refreshed per occurrence."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from data import make_dataset, P, EOS, CONDITIONS
from model import ToyModel


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)


def learning_rate(step,total_steps,warmup=500):
    if step < warmup: return 1e-3*(step+1)/warmup
    progress = (step-warmup)/max(1,total_steps-warmup-1)
    return 1e-3*(0.1+0.9*0.5*(1+math.cos(math.pi*min(progress,1))))


def refresh_noise(arrays,base_n,seed,epoch):
    n = len(arrays['x'])-base_n
    if n == 0: return None
    generator = torch.Generator(device=arrays['x'].device)
    generator.manual_seed(seed*1000000007+epoch*104729+910)
    target = torch.randint(P,(n,),device=arrays['x'].device,generator=generator)
    arrays['y'][base_n:,4] = target
    arrays['x'][base_n:,5] = target
    return target


@torch.no_grad()
def evaluate(model,dataset,device,batch_size=512,save_predictions=None):
    model.eval()
    metrics, predictions = {}, {}
    names = sorted(k[:-7] for k in dataset.files if k.endswith('_answer'))
    for name in names:
        outputs = []
        for start in range(0,len(dataset[name+'_x']),batch_size):
            x = torch.as_tensor(dataset[name+'_x'][start:start+batch_size],device=device)
            pos = torch.as_tensor(dataset[name+'_pos'][start:start+batch_size],device=device)
            pred = model(x)[torch.arange(len(x),device=device),pos].argmax(-1)
            outputs.append(pred.cpu().numpy())
        pred = np.concatenate(outputs)
        gold, first, query = (dataset[name+'_'+s] for s in ['answer','first','query'])
        correct = pred == gold
        wrong_first = (pred == first) & ~correct
        wrong_query = (pred == query) & ~correct & ~wrong_first
        eos = (pred == EOS) & ~correct & ~wrong_first & ~wrong_query
        masks = {'correct':correct,'first':wrong_first,'query':wrong_query,'eos':eos,
                 'other':~(correct|wrong_first|wrong_query|eos)}
        counts = {k:int(v.sum()) for k,v in masks.items()}
        metrics[name] = {'n':len(pred),'counts':counts,'accuracy':float(correct.mean()),
                         'rates':{k:v/len(pred) for k,v in counts.items()}}
        predictions[name] = pred
    for group in ['arithmetic','operator','ordinary']:
        subsets = [v for k,v in metrics.items() if k.startswith(group+'_')]
        n = sum(v['n'] for v in subsets)
        counts = {k:sum(v['counts'][k] for v in subsets) for k in ['correct','first','query','eos','other']}
        metrics[group] = {'n':n,'counts':counts,'accuracy':counts['correct']/n,'rates':{k:v/n for k,v in counts.items()}}
    if save_predictions: np.savez_compressed(save_predictions,**predictions)
    model.train()
    return metrics


def run(args):
    root = Path(args.root).resolve()
    run_dir = root/('smoke' if args.small else 'runs')/f'{args.condition}_seed{args.seed}'
    run_dir.mkdir(parents=True,exist_ok=True)
    lock = (run_dir/'run.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(run_dir/'status.json',{'state':'preparing','condition':args.condition,'seed':args.seed,'target_epochs':args.epochs})
    torch.set_num_threads(args.cpu_threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    device = torch.device(args.device)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(args.seed)
    path = make_dataset(args.seed,root/'datasets',args.small)
    dataset = np.load(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    model = ToyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(0.9,0.999),weight_decay=0.1,fused=device.type=='cuda')
    base_n = len(dataset['base_x'])
    arrays = {k:torch.as_tensor(np.concatenate([dataset['base_'+k],dataset['noise_'+k]])
                if args.condition == 'random_operator' else dataset['base_'+k],device=device)
              for k in ['x','y','w']}
    n = len(arrays['x'])
    source_hashes = {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                     for name in ['data.py','model.py','train.py']}
    config = vars(args)|{'architecture':'transformer','parameters':sum(p.numel() for p in model.parameters()),
        'dataset_metadata':metadata,'source_hashes':source_hashes,'training_examples':n,
        'stable_base_examples':base_n,'random_operator_examples':n-base_n,
        'torch_version':torch.__version__,'numpy_version':np.__version__,
        'precision':'float32 with TF32 allowed','cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device_name':torch.cuda.get_device_name() if device.type=='cuda' else 'cpu'}
    config_path = run_dir/'config.json'
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for key in ['condition','seed','epochs','batch_size','microbatch','small','training_examples','source_hashes']:
            if old[key] != config[key]: raise ValueError(f'Resume mismatch: {key}')
        if old['dataset_metadata']['sha256'] != metadata['sha256']: raise ValueError('Dataset changed')
    atomic_json(config_path,config)
    start_epoch,step,elapsed_before = 0,0,0.0
    checkpoint = run_dir/'latest.pt'
    if checkpoint.exists():
        ckpt = torch.load(checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch,step,elapsed_before = ckpt['epoch'],ckpt['step'],ckpt['elapsed_seconds']
        history_path = run_dir/'history.jsonl'
        if history_path.exists():
            kept = [l for l in history_path.read_text().splitlines() if json.loads(l)['epoch']<=start_epoch]
            history_path.write_text('\n'.join(kept)+ ('\n' if kept else ''))
    steps_per_epoch = math.ceil(n/args.batch_size)
    total_steps = args.epochs*steps_per_epoch
    start_time,latest = time.monotonic(),None
    print(json.dumps({'state':'training','condition':args.condition,'seed':args.seed,'examples':n,
                      'steps_per_epoch':steps_per_epoch,'parameters':config['parameters']}),flush=True)
    for epoch in range(start_epoch,args.epochs):
        epoch_start = time.monotonic()
        refresh_noise(arrays,base_n,args.seed,epoch)
        order = torch.randperm(n,generator=torch.Generator().manual_seed(args.seed*100000+epoch)).to(device)
        epoch_loss = 0.0
        model.train()
        for offset in range(0,n,args.batch_size):
            idx = order[offset:offset+args.batch_size]
            denominator = arrays['w'][idx].sum()
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(step,total_steps)
            for group in optimizer.param_groups: group['lr'] = lr
            step_loss = torch.zeros((),device=device)
            for m in range(0,len(idx),args.microbatch):
                mi = idx[m:m+args.microbatch]
                logits = model(arrays['x'][mi])
                ce = F.cross_entropy(logits.float().flatten(0,1),arrays['y'][mi].flatten(),reduction='none')
                loss = (ce.view_as(arrays['w'][mi])*arrays['w'][mi]).sum()/denominator
                loss.backward();step_loss += loss.detach()
            if not torch.isfinite(step_loss): raise FloatingPointError('Non-finite training loss')
            optimizer.step();epoch_loss += float(step_loss);step += 1
            if offset == 0 or step%10 == 0:
                atomic_json(run_dir/'heartbeat.json',{'epoch':epoch+1,'step':step,'total_steps':total_steps,
                    'epoch_fraction':min(offset+args.batch_size,n)/n,'seconds':elapsed_before+time.monotonic()-start_time,
                    'updated_unix':time.time()})
        row = {'epoch':epoch+1,'step':step,'loss':epoch_loss/steps_per_epoch,'lr':lr,
               'epoch_seconds':time.monotonic()-epoch_start,'elapsed_seconds':elapsed_before+time.monotonic()-start_time}
        if epoch == 0 or (epoch+1)%args.eval_every == 0 or epoch+1 == args.epochs:
            latest = evaluate(model,dataset,device)
            row['evaluation'] = latest
            atomic_json(run_dir/'latest_evaluation.json',{'epoch':epoch+1,'metrics':latest})
        with (run_dir/'history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        atomic_json(run_dir/'status.json',{'state':'training','condition':args.condition,'seed':args.seed,
             'epoch':epoch+1,'target_epochs':args.epochs,'step':step,'latest':row,'last_evaluation':latest})
        print(json.dumps({k:v for k,v in row.items() if k!='evaluation'}),flush=True)
        if (epoch+1)%args.checkpoint_every == 0 or epoch+1 == args.epochs:
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,
                        'step':step,'elapsed_seconds':elapsed_before+time.monotonic()-start_time},checkpoint.with_suffix('.tmp'))
            checkpoint.with_suffix('.tmp').replace(checkpoint)
    metrics = evaluate(model,dataset,device,save_predictions=run_dir/'predictions.npz')
    result = {'condition':args.condition,'seed':args.seed,'completed_epochs':args.epochs,'metrics':metrics,
              'elapsed_seconds':elapsed_before+time.monotonic()-start_time,'small_smoke_test':args.small}
    atomic_json(run_dir/'result.json',result)
    atomic_json(run_dir/'status.json',{'state':'complete','epoch':args.epochs,'target_epochs':args.epochs})
    print('COMPLETE '+str(run_dir),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',default=str(Path(__file__).parent))
    p.add_argument('--condition',choices=CONDITIONS,required=True)
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--epochs',type=int,default=1200)
    p.add_argument('--batch-size',type=int,default=8190)
    p.add_argument('--microbatch',type=int,default=2048)
    p.add_argument('--eval-every',type=int,default=25)
    p.add_argument('--checkpoint-every',type=int,default=10)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--small',action='store_true')
    args = p.parse_args()
    if min(args.epochs,args.batch_size,args.microbatch,args.eval_every,args.checkpoint_every,args.cpu_threads)<=0:
        p.error('Numeric training settings must be positive')
    run(args)
