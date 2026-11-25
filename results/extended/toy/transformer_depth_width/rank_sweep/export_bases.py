from pathlib import Path
import torch,json,hashlib
root=Path(__file__).resolve().parent
torch.set_num_threads(3)
for w in (256,1024):
 source=root/f'w{w}/bases.pt'
 if not source.exists():continue
 data=torch.load(source,weights_only=True)
 dest=root/f'w{w}/directions';dest.mkdir(exist_ok=True)
 for layer,basis in data.items():
  f=dest/f'layer{layer+1:02}.pt'
  if not f.exists():torch.save({k:v.clone() for k,v in basis.items()},f)
proto=json.loads((root.parent/'subspace_intervention/protocol.json').read_text())
proto.pop('ranks',None)
proto.update(rank_grid=[1,4,16,64,128,256],arithmetic_directions='independent SVD of training gradients; no projection against identity directions',random_controls=2,expected_conditions=540)
(root/'protocol.json').write_text(json.dumps(proto,indent=2)+'\n')
