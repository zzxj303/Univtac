from adapters.datasets import resolve_episode
import os
"""Train-only quantiles, fixed episode split, next-qpos delta chunks and masks."""
import json
from types import SimpleNamespace
import h5py,numpy as np,torch,sentencepiece
from torch.utils.data import Dataset,DataLoader
from openpi_client.image_tools import resize_with_pad
from openpi.models.tokenizer import PaligemmaTokenizer
from common import ROOT,OUT,PROMPTS,keys,atomic_json,seed_worker

def split_for(task):
 return json.loads(open(os.environ['TH_SPLIT']).read())

def targets(q,a,t):
 x=a[t:t+50,:8].copy();x[:,:7]-=q[t,:7]
 return x

def stats_for(task):
 return json.loads(open(os.environ['TH_STATS']).read())

def normalize(x,s):return (2*(x-np.asarray(s['q01']))/(np.asarray(s['q99'])-np.asarray(s['q01'])+1e-6)-1).astype('float32')
def unnormalize(x,s):return (x+1)/2*(np.asarray(s['q99'])-np.asarray(s['q01'])+1e-6)+np.asarray(s['q01'])
def tokenizer():
 tok=PaligemmaTokenizer.__new__(PaligemmaTokenizer);tok._max_len=200
 tok._tokenizer=sentencepiece.SentencePieceProcessor(model_file=str(ROOT.parent/'rollout/.cache/openpi/big_vision/paligemma_tokenizer.model'))
 return tok

def observation(images,q,task,mode,stats,tok):
 state=np.zeros(32,dtype='float32');state[:8]=normalize(q[:8],stats['state'])
 tokens,mask=tok.tokenize(PROMPTS[task],state=state[:8])
 ims={k:torch.from_numpy(resize_with_pad(images[k],224,224).copy()).permute(2,0,1).float()/127.5-1 for k in keys(mode)}
 return {'images':ims,'state':torch.from_numpy(state),'tokens':torch.from_numpy(tokens).long(),'token_mask':torch.from_numpy(mask).bool()}

class Episodes(Dataset):
 def __init__(self,task,mode,partition='train_ids'):
  self.task=task;self.mode=mode;self.stats=stats_for(task);self.tokenizer=tokenizer();self.directory=__import__('pathlib').Path(os.environ['TH_DATA_DIR'])
  self.index=[];self.q={};self.a={}
  for i in split_for(task)[partition]:
   with h5py.File(resolve_episode(self.directory,i)) as f:self.q[i]=f['observations/qpos'][:,:8];self.a[i]=f['action'][:,:8]
   self.index.extend((i,t) for t in range(len(self.a[i])))
 def __len__(self):return len(self.index)
 def __getitem__(self,index):
  i,t=self.index[index];q=self.q[i][t]
  with h5py.File(resolve_episode(self.directory,i)) as f:ims={k:f['observations/images/'+k][t] for k in keys(self.mode)}
  d=observation(ims,q,self.task,self.mode,self.stats,self.tokenizer)
  x=targets(self.q[i],self.a[i],t);n=len(x);a=np.zeros((50,32),dtype='float32');mask=np.zeros((50,32),dtype=bool)
  a[:n,:8]=normalize(x,self.stats['action']);mask[:n,:8]=True
  d.update(actions=torch.from_numpy(a),valid=torch.from_numpy(mask),index=index,qpos=torch.from_numpy(q.copy()))
  return d

class StatefulSampler(torch.utils.data.Sampler):
 def __init__(self,n):self.n=n;self.generator=torch.Generator().manual_seed(42);self.order=[];self.cursor=0;self.epoch=0
 def __iter__(self):
  while True:
   if self.cursor==len(self.order):self.order=torch.randperm(self.n,generator=self.generator).tolist();self.cursor=0;self.epoch+=1
   value=self.order[self.cursor];self.cursor+=1;yield value
 def state_dict(self):return {'generator':self.generator.get_state(),'order':self.order,'cursor':self.cursor,'epoch':self.epoch,'n':self.n}
 def load_state_dict(self,s):
  assert s['n']==self.n
  self.generator.set_state(s['generator']);self.order=s['order'];self.cursor=s['cursor'];self.epoch=s['epoch']

def loader(dataset,micro=1):
 sampler=StatefulSampler(len(dataset));generator=torch.Generator().manual_seed(42)
 # No prefetch: sampler cursor always identifies the next unconsumed sample for exact resume.
 dl=DataLoader(dataset,batch_size=micro,sampler=sampler,num_workers=0,worker_init_fn=seed_worker,generator=generator,pin_memory=True)
 return dl,sampler,generator

def to_device(batch,device='cuda'):
 obs=SimpleNamespace(images={k:v.to(device,non_blocking=True) for k,v in batch['images'].items()},image_masks={k:torch.ones(len(batch['state']),dtype=torch.bool,device=device) for k in batch['images']},state=batch['state'].to(device),tokenized_prompt=batch['tokens'].to(device),tokenized_prompt_mask=batch['token_mask'].to(device))
 return obs,batch['actions'].to(device),batch['valid'].to(device)
