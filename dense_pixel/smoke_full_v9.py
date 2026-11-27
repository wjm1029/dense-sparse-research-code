"""Exercise full-dataset reader, statistics, gradients, and folded checkpoint replay."""
import hashlib
import json
import torch
from torch import nn
import train_full_v9 as full
from full_data_v9 import RUN,atomic
from train_multiclip_v8 import Branch,save_atomic,infer,folded_infer

torch.set_num_threads(4);torch.manual_seed(20261130)
torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
records=[json.loads(p.read_text()) for p in sorted((RUN/'processed').glob('*.json'))]
rows=[r['row'] for r in records if r['status']=='ready' and r['row']['split']=='train'][:4]
assert len(rows)==4
full.OUT=RUN/'smoke_checks';full.OUT.mkdir(exist_ok=True)
fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
stats=full.fit_stats(rows,'depth',fingerprint)
heads=nn.ModuleDict({n:Branch(s,2048,.05) for n,s in stats.items()}).cuda()
bank=[full.prepare_case(heads,full.load_case(r,'depth')) for r in rows]
optimizer=torch.optim.AdamW(heads.parameters(),lr=3e-4,weight_decay=.001)
initial=heads['regular'].net[-1].weight.detach().clone()
losses=[]
for step in range(8):
    optimizer.zero_grad(set_to_none=True);total=0.
    for name,batch,weight in [('first',64,1/81),('regular',512,80/81)]:
        xs=[];ys=[]
        for b in bank:
            x,y=b[name];i=torch.randint(len(x),(batch,),device='cuda');xs.append(x[i]);ys.append(y[i])
        loss=(heads[name].net(torch.cat(xs))-torch.cat(ys)).square().mean()*weight
        assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
    optimizer.step();losses.append(total)
assert not torch.equal(initial,heads['regular'].net[-1].weight)
exported={n:h.export() for n,h in heads.items()}
save_atomic(exported,full.OUT/'folded_head.pt')
reloaded=torch.load(full.OUT/'folded_head.pt',map_location='cpu',weights_only=True)
case=full.load_case(rows[0],'depth')
with torch.no_grad():
    a=infer(heads,case['features']);b=folded_infer(reloaded,case['features'])
    difference=float((a-b).abs().max());assert difference<.001,difference
receipt=dict(status='passed',scope='4 real cached videos used only to test streaming reader/statistics/training/checkpoint code',
    samples=[r['sample'] for r in rows],gradient_steps=8,losses=losses,folded_max_difference=difference,
    full_training_started=False)
atomic(full.OUT/'COMPLETE.json',receipt);print(json.dumps(receipt),flush=True)
