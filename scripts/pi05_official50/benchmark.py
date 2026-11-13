"""Measure real forward/backward micro-batches before selecting accumulation."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse,time,gc
import torch
from common import *
from model import Policy,MasterAdamW
from data import Episodes,loader,to_device
from train import flow_random
p=argparse.ArgumentParser();p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--structure-only',action='store_true');a=p.parse_args()
seed_all();model=Policy(a.mode,load_base=not a.structure_only).cuda();model.train();opt=MasterAdamW(model);d=Episodes('insert_HDMI',a.mode);generator=torch.Generator(device='cuda').manual_seed(42)
rows=[]
for micro in [1,2,4]:
 dl,_,_=loader(d,micro);batch=next(iter(dl));obs,actions,valid=to_device(batch);torch.cuda.reset_peak_memory_stats();times=[]
 try:
  for repeat in range(3):
   opt.zero_grad();noise,t=flow_random(actions,generator);torch.cuda.synchronize();start=time.time()
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=(model(obs,actions,noise=noise,time=t).float()*valid).sum()/valid.sum()
   assert torch.isfinite(loss);loss.backward()
   if repeat==0:
    missing=[n for n,p in model.trainables().items() if p.grad is None];assert not missing,missing
   opt.step(1);torch.cuda.synchronize();times.append(time.time()-start)
  rows.append({'micro':micro,'seconds_per_sample':sum(times[1:])/2/micro,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'loss':float(loss)})
  print(rows[-1],flush=True)
 except torch.cuda.OutOfMemoryError:
  rows.append({'micro':micro,'oom':True});break
 opt.zero_grad();del obs,actions,valid,batch;gc.collect();torch.cuda.empty_cache()
atomic_json(OUT/'setup'/f'benchmark_{a.mode}{"_structure" if a.structure_only else ""}.json',rows)
