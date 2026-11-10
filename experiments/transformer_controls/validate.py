"""Validate the scientific controls, target shifting, and model causality."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import numpy as np
import torch
from data import make_dataset,EXCLUDED,SEP,P,ORDINARY,OPS
from model import ToyModel
from train import refresh_noise


def main(args):
    root = Path(__file__).parent
    torch.set_num_threads(2)
    result = {'model':{},'datasets':{}}
    torch.manual_seed(123)
    model = ToyModel().eval()
    spec = importlib.util.spec_from_file_location('original_model',root/'reference_20260912/model.py')
    original = importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    torch.manual_seed(123)
    reference = original.ToyModel('transformer').eval()
    assert model.state_dict().keys()==reference.state_dict().keys()
    assert all(torch.equal(v,reference.state_dict()[k]) for k,v in model.state_dict().items())
    x = torch.randint(0,331,(3,11))
    with torch.no_grad():
        y = model(x)
        assert torch.equal(y,reference(x))
        changed = x.clone();changed[:,5:]=torch.randint(0,331,changed[:,5:].shape)
        assert torch.equal(y[:,:5],model(changed)[:,:5]),'Future target leaked into prediction'
    assert not any(isinstance(m,torch.nn.Conv1d) for m in model.modules())
    assert sum(p.numel() for p in model.parameters())==3331920
    result['model']={'same_parameters_and_outputs_as_original_plain_transformer':True,
                     'causal_target_position_check':True,'parameters':3331920,'hybrid_layers':0}
    for seed in args.seeds:
        path = make_dataset(seed,root/'datasets',small=not args.full)
        z = np.load(path)
        meta = json.loads(path.with_suffix('.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
        x,y = z['base_x'],z['base_y']
        assert np.array_equal(x[:,1:],y[:,:-1])
        ret = x[meta['stable_arithmetic_examples']:]
        sep_idx = (ret==SEP).argmax(1)
        for a,b in EXCLUDED:
            mem_pairs = ((ret[:,:-1]==a)&(ret[:,1:]==b))
            for j in range(ret.shape[1]-1):
                assert not np.any(mem_pairs[:,j]&(j>=1)&(j+1<sep_idx))
        for train,test in zip(z['train_operand_ids'],z['held_operand_ids']): assert not np.intersect1d(train,test).size
        assert not np.intersect1d(z['noise_train_operand_ids'],z['noise_held_operand_ids']).size
        base_n = len(x)
        arrays = {k:torch.from_numpy(np.concatenate([z['base_'+k],z['noise_'+k]])) for k in ['x','y','w']}
        original_base = arrays['x'][:base_n].clone()
        a = refresh_noise(arrays,base_n,seed,0).clone()
        assert torch.equal(a,refresh_noise(arrays,base_n,seed,0))
        b = refresh_noise(arrays,base_n,seed,1).clone()
        change_rate = float((a!=b).float().mean())
        assert change_rate>0.97
        assert torch.equal(arrays['x'][base_n:,5],arrays['y'][base_n:,4])
        assert torch.equal(arrays['x'][:base_n],original_base)
        assert torch.equal(arrays['x'][:,1:],arrays['y'][:,:-1])
        pair_names = ['operator_'+f'{a}_{b}' for a,b in OPS]+['ordinary_'+f'{a}_{b}' for a,b in ORDINARY]
        canonical = None
        for name,(a,b) in zip(pair_names,OPS+ORDINARY):
            tx,ty,pos = z[name+'_x'],z[name+'_y'],z[name+'_pos']
            assert np.array_equal(ty[np.arange(len(ty)),pos],z[name+'_answer'])
            assert np.all(z[name+'_answer']==b) and np.all(z[name+'_first']==a)
            stripped = []
            for row,p in zip(tx,pos):
                sep = list(row).index(SEP)
                mem = list(row[1:sep]); query = int(row[p]);j = mem.index(query)
                assert mem[j-2:j]==[a,b]
                stripped.append((tuple(mem[:j-2]+mem[j:]),j,int(query)))
            if canonical is None:canonical=stripped
            else:assert stripped==canonical,'Contexts differ across pair evaluations'
        result['datasets'][str(seed)]={'sha256':meta['sha256'],'full':args.full,
             'excluded_pair_training_occurrences':meta['retrieval_audit']['excluded_memory_pair_counts'],
             'random_label_change_fraction_between_epochs':change_rate,'deterministic_resume_labels':True,
             'shifted_input_target_consistency':True,'shared_pair_test_contexts':True,
             'base_examples':base_n,'noise_examples':len(z['noise_x'])}
        print('PASS dataset',seed,'full' if args.full else 'small',flush=True)
    out = root/('validation_full.json' if args.full else 'validation_small.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS model and causality; report:',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');p.add_argument('--seeds',nargs='+',type=int,default=[101,102,103])
    main(p.parse_args())
