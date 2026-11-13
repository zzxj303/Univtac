"""Two/four-view upstream PI05 with BF16 VLM LoRA and FP32 full expert."""
import math
import torch
from torch import nn
from safetensors.torch import load_model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from common import BASE,keys,RECIPE

class LoRALinear(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base;self.in_features=base.in_features;self.out_features=base.out_features
  self.A=nn.Parameter(torch.empty(16,self.in_features,dtype=torch.bfloat16,device=base.weight.device));self.B=nn.Parameter(torch.zeros(self.out_features,16,dtype=torch.bfloat16,device=base.weight.device))
  nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
 @property
 def weight(self):return self.base.weight
 @property
 def bias(self):return self.base.bias
 def forward(self,x):return self.base(x)+torch.nn.functional.linear(torch.nn.functional.linear(x,self.A),self.B)*2

class Policy(PI0Pytorch):
 def __init__(self,mode,load_base=True):
  super().__init__(Pi0Config(pi05=True,action_horizon=50));self.mode=mode
  # Retain eager upstream inference; compilation is not needed for short offline checks.
  self.sample_actions=PI0Pytorch.sample_actions.__get__(self,type(self))
  self.paligemma_with_expert.gemma_expert.lm_head=nn.Identity()
  self.float();self.paligemma_with_expert.paligemma.to(torch.bfloat16)
  if load_base:load_model(self,str(BASE),strict=True)
  for p in self.parameters():p.requires_grad_(False)
  vlm=self.paligemma_with_expert.paligemma
  # In the final joint layer, prefix queries/output/MLP cannot influence the suffix.
  last=len(vlm.language_model.layers)-1
  unused=(f'model.language_model.layers.{last}.self_attn.q_proj',f'model.language_model.layers.{last}.self_attn.o_proj',f'model.language_model.layers.{last}.mlp.')
  names=[name for name,module in vlm.named_modules() if isinstance(module,nn.Linear) and name!='lm_head' and not any(name.startswith(x) for x in unused)]
  for name in names:
   parent,_,child=name.rpartition('.');setattr(vlm.get_submodule(parent),child,LoRALinear(vlm.get_submodule(name)))
  for name,p in self.named_parameters():
   if not name.startswith('paligemma_with_expert.paligemma.') and not name.startswith('paligemma_with_expert.gemma_expert.lm_head.'):
    p.requires_grad_(True)
  self.gradient_checkpointing_enable()
  torch.set_float32_matmul_precision("highest")
 def _preprocess_observation(self,observation,train=True):
  expected=keys(self.mode);assert list(observation.images)==expected
  return [observation.images[k] for k in expected],[observation.image_masks[k] for k in expected],observation.tokenized_prompt,observation.tokenized_prompt_mask,observation.state
 def trainables(self):return {n:p for n,p in self.named_parameters() if p.requires_grad}
 def audit(self):
  rows=[]
  for name,p in self.named_parameters():
   vlm=name.startswith('paligemma_with_expert.paligemma.')
   if vlm:
    assert p.dtype==torch.bfloat16,(name,p.dtype)
    assert p.requires_grad==name.endswith(('.A','.B')),name
   elif not '.lm_head.' in name:assert p.requires_grad and p.dtype==torch.float32,name
   rows.append({'name':name,'shape':list(p.shape),'dtype':str(p.dtype),'trainable':p.requires_grad,'numel':p.numel()})
  return rows
 def load_trainables(self,state):
  current=self.trainables();assert set(current)==set(state)
  with torch.no_grad():
   for n,p in current.items():assert p.shape==state[n].shape;p.copy_(state[n])

def lr_multiplier(step):
 total=RECIPE['steps'];warmup=RECIPE['warmup'];minimum=RECIPE['min_lr_ratio']
 assert 1<=step<=total and 0<warmup<total
 return step/warmup if step<=warmup else minimum+(1-minimum)*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))

class MasterAdamW:
 """FP32 optimizer masters for BF16 adapters; experts already have FP32 parameters."""
 def __init__(self,model):
  self.named=model.trainables();self.masters={n:nn.Parameter(p.detach().float().clone()) for n,p in self.named.items() if p.dtype==torch.bfloat16}
  groups=[{'params':list(self.masters.values()),'lr':1e-4,'peak_lr':1e-4},{'params':[p for n,p in self.named.items() if n not in self.masters],'lr':5e-5,'peak_lr':5e-5}]
  self.optimizer=torch.optim.AdamW(groups,betas=(.9,.95),weight_decay=.01)
 def zero_grad(self):
  self.optimizer.zero_grad(set_to_none=True)
  for p in self.named.values():p.grad=None
 def step(self,step):
  for n,p in self.masters.items():p.grad=None if self.named[n].grad is None else self.named[n].grad.float()
  params=[p for g in self.optimizer.param_groups for p in g['params']]
  norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
  factor=lr_multiplier(step)
  for g in self.optimizer.param_groups:g['lr']=g['peak_lr']*factor
  self.optimizer.step()
  with torch.no_grad():
   for n,p in self.masters.items():self.named[n].copy_(p)
  return float(norm)
 def state_dict(self):return {'optimizer':self.optimizer.state_dict(),'masters':{n:p.detach().cpu() for n,p in self.masters.items()}}
 def load_state_dict(self,state):
  self.optimizer.load_state_dict(state['optimizer'])
  with torch.no_grad():
   for n,p in self.masters.items():p.copy_(state['masters'][n]);self.named[n].copy_(p)
