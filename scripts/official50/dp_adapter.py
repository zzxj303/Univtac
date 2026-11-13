"""UniVTAC observation adapter around the upstream ViTAL diffusion network/train loop."""
import json,sys
from pathlib import Path
import h5py,numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'policy/ViTAL/diffusion'
sys.path.insert(0,str(UPSTREAM))
from network import get_resnet,replace_bn_with_gn,ConditionalUnet1D

class Dataset50(Dataset):
 def __init__(self,task,ids,stats,tactile):
  self.path=ROOT/'data/official50'/task;self.ids=ids;self.stats=stats;self.tactile=tactile
  self.rgb=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);self.tac=transforms.ToTensor()
 def __len__(self): return len(self.ids)
 def __getitem__(self,index):
  with h5py.File(self.path/f'episode_{self.ids[index]}.hdf5') as f:
   n=len(f['action']);t=np.random.randint(n)
   action=f['action'][t:min(t+20,n)]
   action=np.concatenate([action,np.repeat(action[-1:],20-len(action),axis=0)])
   q=f['observations/qpos'][t]
   s=self.stats
   action=2*(action-s['action_min'])/np.maximum(s['action_max']-s['action_min'],1e-6)-1
   q=2*(q-s['qpos_min'])/np.maximum(s['qpos_max']-s['qpos_min'],1e-6)-1
   d={'action':torch.tensor(action,dtype=torch.float32),'agent_pos':torch.tensor(q,dtype=torch.float32)[None]}
   for name in ['cam_high','cam_wrist']:
    d[name]=self.rgb(f['observations/images/'+name][t])[None]
   if self.tactile:
    for name in ['tac_left','tac_right']:d[name]=self.tac(f['observations/images/'+name][t])[None]
   return d

def stats_for(task):
 q=[];a=[]
 for i in range(50):
  with h5py.File(ROOT/'data/official50'/task/f'episode_{i}.hdf5') as f:q.append(f['observations/qpos'][:]);a.append(f['action'][:])
 q=np.concatenate(q);a=np.concatenate(a)
 return {'qpos_min':q.min(0),'qpos_max':q.max(0),'action_min':a.min(0),'action_max':a.max(0)}

def create(task,mode):
 tactile=mode=='rgb_tactile';names=['cam_high','cam_wrist']+(['tac_left','tac_right'] if tactile else [])
 stats=stats_for(task);split=json.loads((ROOT/'data/official50'/task/'split.json').read_text())
 nets=nn.ModuleDict({'noise_pred_net':ConditionalUnet1D(input_dim=8,global_cond_dim=512*len(names)+8)})
 for name in names:nets[name+'_encoder']=replace_bn_with_gn(get_resnet('resnet18'))
 datasets=[Dataset50(task,split[k],stats,tactile) for k in ['train_ids','validation_ids']]
 loaders=[DataLoader(d,batch_size=bs,num_workers=n,shuffle=True,pin_memory=True,persistent_workers=True,worker_init_fn=seed_worker) for d,bs,n in zip(datasets,[8,2],[4,1])]
 return nets,loaders,names,stats,split

def seed_worker(worker_id):
 import random
 seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed)
