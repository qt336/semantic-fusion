from pathlib import Path
import json,hashlib
import torch
root=Path(__file__).resolve().parent
out={'seed':106,'ranks':[16],'arithmetic_training_examples':768,'random_controls':2,'torch':torch.__version__,'models':{}}
for width in (256,1024):
 p=root/f'w{width}'
 ck=torch.load(p/'backbone.pt',weights_only=True,map_location='cpu',mmap=True)
 out['models'][width]={'epoch':ck['epoch'],'step':ck['step'],'layers':8,'heads':8,'d_model':width,'checkpoint_sha256':hashlib.sha256((p/'backbone.pt').read_bytes()).hexdigest()}
base=Path('./source_L8_D1024')
out['data_sha256']=hashlib.sha256((base/'datasets/seed_106.npz').read_bytes()).hexdigest()
out['probe_data_sha256']=hashlib.sha256((base/'all_layers_probe_seed106/ordinary.npz').read_bytes()).hexdigest()
(root/'protocol.json').write_text(json.dumps(out,indent=2)+'\n')
