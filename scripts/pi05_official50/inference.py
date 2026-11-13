"""Offline/online callable adapter. No simulator is started by this module."""
import numpy as np,torch
from torch.utils.data import default_collate
from common import seed_all,keys,OUT,BASE,sha256
from data import observation,tokenizer,unnormalize,to_device
from model import Policy

class Pi05Controller:
 def __init__(self,checkpoint,device='cuda'):
  seed_all();state=torch.load(checkpoint,map_location='cpu',weights_only=False);self.config=state['config'];self.device=device
  import json
  from pathlib import Path
  assert sha256(BASE)==self.config['base_sha256'],'Base checkpoint mismatch'
  assert sha256(Path(checkpoint).parent/'stats.json')==self.config['stats_sha256'],'Normalization statistics mismatch'
  self.stats=json.loads((Path(checkpoint).parent/'stats.json').read_text());self.tok=tokenizer()
  self.model=Policy(self.config['mode']);self.model.load_trainables(state['weights']);self.model.to(device).eval();self.reset()
 def reset(self,seed=42):self.generator=torch.Generator(device=self.device).manual_seed(seed)
 @torch.no_grad()
 def predict(self,images,qpos):
  """images use official50 cam_high/cam_wrist/tac_left/tac_right channel order."""
  d=observation(images,qpos,self.config['task'],self.config['mode'],self.stats,self.tok)
  d.update(actions=torch.zeros(50,32),valid=torch.ones(50,32,dtype=torch.bool));obs,_,_=to_device(default_collate([d]),self.device)
  noise=torch.randn((1,50,32),generator=self.generator,device=self.device)
  with torch.autocast('cuda',dtype=torch.bfloat16):a=self.model.sample_actions(self.device,obs,noise=noise,num_steps=10)
  a=unnormalize(a[0,:,:8].float().cpu().numpy(),self.stats['action']);a[:,:7]+=np.asarray(qpos)[:7]
  if not np.isfinite(a).all():raise FloatingPointError('Nonfinite predicted action')
  return a.astype('float32')
 def act(self,env,images,qpos):
  action=self.predict(images,qpos)[0]
  env.take_action(action,action_type='qpos')
  return action
