import json,random
from pathlib import Path
import numpy as np,torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from dp_adapter import ROOT,create

torch.set_num_threads(4)
report={'datasets':{},'dp':{}}
for task in ['insert_HDMI','insert_hole','pull_out_key']:
 p=ROOT/'data/official50'/task;s=json.loads((p/'split.json').read_text())
 assert len(s['train_ids'])==40 and len(s['validation_ids'])==10
 assert set(s['train_ids']).isdisjoint(s['validation_ids'])
 assert sorted(s['train_ids']+s['validation_ids'])==list(range(50))
 assert len(list(p.glob('episode_*.hdf5')))==50
 report['datasets'][task]={'episodes':50,'train':40,'validation':10}
for mode in ['rgb','rgb_tactile']:
 torch.manual_seed(42);np.random.seed(42);random.seed(42)
 nets,loaders,names,stats,split=create('insert_HDMI',mode)
 ckpt=torch.load(ROOT/'outputs/official50/smoke_dp/insert_HDMI'/mode/'policy_best.ckpt',map_location='cpu',weights_only=False)
 changed={name:bool(torch.any(nets[name+'_encoder'].conv1.weight != ckpt['state_dict'][name+'_encoder.conv1.weight'])) for name in names}
 assert all(changed.values()),changed
 nets.load_state_dict(ckpt['state_dict'],strict=True);nets.cuda().eval()
 batch=loaders[1].dataset[0]
 with torch.inference_mode():
  feats=[nets[name+'_encoder'](batch[name].cuda()) for name in names]
  cond=torch.cat(feats+[batch['agent_pos'].cuda()],dim=-1)
  scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2',clip_sample=True,prediction_type='epsilon');scheduler.set_timesteps(100)
  action=torch.randn(1,20,8,device='cuda')
  for t in scheduler.timesteps:
   noise=nets['noise_pred_net'](action,t,global_cond=cond)
   action=scheduler.step(noise,t,action).prev_sample
  assert action.shape==(1,20,8) and torch.isfinite(action).all()
 report['dp'][mode]={'encoder_weights_updated':changed,'prediction_shape':list(action.shape),'finite':True,'denoising_steps':100}
 del nets,loaders,ckpt
 torch.cuda.empty_cache()
(ROOT/'outputs/official50/preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
