"""Check the approved 4000-step LR boundaries and optimizer replay across warmup."""
import io,math
import torch
from common import RECIPE,OUT,atomic_json
from model import lr_multiplier,MasterAdamW
assert RECIPE['steps']==4000 and RECIPE['warmup']==200
v=[lr_multiplier(i) for i in range(1,4001)]
assert v[0]==.005 and v[199]==1 and math.isclose(v[-1],.1)
assert all(a<=b for a,b in zip(v[:199],v[1:200]))
assert all(a>=b for a,b in zip(v[199:-1],v[200:]))
try:lr_multiplier(4001)
except AssertionError:pass
else:raise AssertionError('Exceeded budget accepted')
class Tiny:
 def __init__(self):self.params={'lora':torch.nn.Parameter(torch.tensor([.2,.3],dtype=torch.bfloat16)),'expert':torch.nn.Parameter(torch.tensor([.4,.5],dtype=torch.float32))}
 def trainables(self):return self.params
m=Tiny();opt=MasterAdamW(m)
def update(step):
 opt.zero_grad()
 for p in m.params.values():p.grad=torch.ones_like(p)*.1
 opt.step(step)
update(199);update(200)
b=io.BytesIO();torch.save({'params':{n:p.detach().clone() for n,p in m.params.items()},'optimizer':opt.state_dict()},b)
update(201);expected={n:p.detach().clone() for n,p in m.params.items()};lrs=[g['lr'] for g in opt.optimizer.param_groups]
b.seek(0);state=torch.load(b,weights_only=False)
with torch.no_grad():
 for n,p in m.params.items():p.copy_(state['params'][n])
opt.load_state_dict(state['optimizer']);update(201)
assert all(torch.equal(p,expected[n]) for n,p in m.params.items())
assert lrs==[g['lr'] for g in opt.optimizer.param_groups]
update(4000)
assert math.isclose(opt.optimizer.param_groups[0]['lr'],1e-5)
assert math.isclose(opt.optimizer.param_groups[1]['lr'],5e-6)
atomic_json(OUT/'setup/schedule_4000_audit.json',dict(passed=True,steps=4000,warmup=200,first_multiplier=v[0],peak_multiplier=v[199],final_multiplier=v[-1],optimizer_resume_exact=True))
print('4000-step schedule and optimizer replay passed')
