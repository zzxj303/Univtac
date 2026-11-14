"""Independent unequal-length example catches per-window averaging and padding leakage."""
import torch
from common import OUT,atomic_json
from train import validate
class FixedLoss(torch.nn.Module):
 def forward(self,obs,actions,noise=None,time=None):return actions
samples=[]
for length,value in [(50,1.),(1,9.)]:
 values=torch.full((50,32),1e6);mask=torch.zeros(50,32,dtype=torch.bool)
 values[:length,:8]=value;mask[:length,:8]=True
 samples.append(dict(images={'cam_high':torch.zeros(3,224,224),'cam_wrist':torch.zeros(3,224,224)},state=torch.zeros(32),tokens=torch.zeros(200,dtype=torch.long),token_mask=torch.ones(200,dtype=torch.bool),actions=values,valid=mask))
expected=(400*1+8*9)/408
results=[validate(FixedLoss(),samples,micro) for micro in (1,2)]
assert all(r['elements']==408 and r['loss']==expected for r in results),results
atomic_json(OUT/'setup/weighting_audit.json',dict(passed=True,expected=expected,results=results,padding_loss=1e6))
print('WEIGHTING_AND_PADDING_PASS',results)
