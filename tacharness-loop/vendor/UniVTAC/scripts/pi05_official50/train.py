"""Fixed seed-42 training, element-weighted validation and exact resumable state."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse,contextlib,fcntl,json,math,random,shutil,time,traceback
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from common import *
from data import Episodes,loader,to_device,split_for
from model import Policy,MasterAdamW


def disk_check(extra=0):
 free=shutil.disk_usage(OUT).free
 if free-extra<8*1024**3:raise RuntimeError(f'Disk reserve violation: free={free}, required_write={extra}, reserve=8 GiB')

def save_torch(path,state):
 path=Path(path);temp=path.with_suffix('.tmp');disk_check()
 with temp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
 disk_check();os.replace(temp,path)

def random_state(noise,sampler,dlgen):
 return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),noise=noise.get_state(),sampler=sampler.state_dict(),loader=dlgen.get_state())

def restore_random(s,noise,sampler,dlgen):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda']);noise.set_state(s['noise']);sampler.load_state_dict(s['sampler']);dlgen.set_state(s['loader'])

def flow_random(actions,generator):
 noise=torch.randn(actions.shape,device=actions.device,generator=generator)
 time_=torch.rand((actions.shape[0],),device=actions.device,generator=generator).pow(1/1.5)*.999+.001
 return noise,time_

def update(model,opt,iterator,generator,step,micro=1,audit=False):
 batches=[next(iterator) for _ in range(32//micro)]
 denominator=sum(int(b['valid'].sum()) for b in batches)
 opt.zero_grad();value=0.;indices=[]
 for batch in batches:
  obs,actions,valid=to_device(batch);noise,t=flow_random(actions,generator)
  with torch.autocast('cuda',dtype=torch.bfloat16):losses=model(obs,actions,noise=noise,time=t)
  loss=(losses.float()*valid).sum()/denominator
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
  loss.backward();value+=float(loss.detach());indices.extend(batch['index'].tolist())
 grad_audit={}
 if audit:
  for n,p in model.trainables().items():
   if p.grad is None:raise AssertionError('No gradient: '+n)
   if not torch.isfinite(p.grad).all():raise AssertionError('Nonfinite gradient: '+n)
   grad_audit[n]=float(p.grad.float().norm())
 norm=opt.step(step)
 return {'loss':value,'grad_norm':norm,'indices':indices,'gradients':grad_audit}

@torch.no_grad()
def validate(model,dataset,micro=1):
 # Re-created fixed stream makes validation comparable and cannot consume train RNG.
 generator=torch.Generator(device='cuda').manual_seed(42)
 dlgen=torch.Generator().manual_seed(42)
 dl=DataLoader(dataset,batch_size=micro,shuffle=False,num_workers=0,generator=dlgen)
 model.eval();total=0.;count=0
 for batch in dl:
  obs,actions,valid=to_device(batch);noise,t=flow_random(actions,generator)
  with torch.autocast('cuda',dtype=torch.bfloat16):losses=model(obs,actions,noise=noise,time=t)
  total+=float((losses.float()*valid).sum());count+=int(valid.sum())
 if not math.isfinite(total):raise FloatingPointError('Nonfinite validation loss')
 model.train();return {'loss':total/count,'elements':count,'samples':len(dataset)}

def checkpoint(directory,model,opt,step,best,generator,sampler,dlgen,config,is_best=False):
 # Serialize large writes across the four GPU jobs to bound temporary disk usage.
 with (OUT/"checkpoint.lock").open("w") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _checkpoint_unlocked(directory,model,opt,step,best,generator,sampler,dlgen,config,is_best)

def _checkpoint_unlocked(directory,model,opt,step,best,generator,sampler,dlgen,config,is_best=False):
 weights={n:p.detach().cpu().clone() for n,p in model.trainables().items()}
 bytes_=sum(p.numel()*p.element_size() for p in weights.values())
 disk_check(bytes_*4+1024**3)
 # The complete recovery file is authoritative if a crash splits the last-model writes.
 recovery=dict(weights=weights,optimizer=opt.state_dict(),step=step,best=best,is_best=is_best,random=random_state(generator,sampler,dlgen),config=config)
 save_torch(directory/'recovery.pt',recovery)
 artifact=dict(weights=weights,step=step,config=config,best=best)
 save_torch(directory/'last.pt',artifact)
 if is_best:save_torch(directory/'best.pt',artifact)

def run(task,mode,directory,resume=False):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 with (directory/'run.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  status=directory/'status.json'
  if (directory/'last.pt').exists() and not resume:raise RuntimeError('Output already contains a run; use --resume')
  acceptance=json.loads((OUT/'acceptance.json').read_text());assert acceptance['passed'], 'Offline acceptance is required'
  seed_all();manifest=json.loads((OUT/'base/manifest.json').read_text())
  assert sha256(BASE)==manifest['sha256'],'Base checkpoint hash mismatch'
  for name,digest in acceptance['script_hashes'].items():assert sha256(Path(__file__).parent/name)==digest,'Accepted source changed: '+name
  config=dict(RECIPE,task=task,mode=mode,base_sha256=manifest['sha256'],image_keys=keys(mode),prompt=PROMPTS[task],split=split_for(task),stats_sha256=sha256(OUT/'stats'/f'{task}.json'))
  # Measured single-run micro-batch from the acceptance report, always preserving batch 32.
  micro=acceptance['micro_batch'];assert micro==RECIPE['micro_batch']==8
  assert RECIPE['accumulation']==32//micro
  config.update(micro_batch=micro,accumulation=32//micro)
  atomic_json(directory/'config.json',config);atomic_json(directory/'versions.json',versions())
  train=Episodes(task,mode);val=Episodes(task,mode,'validation_ids');atomic_json(directory/'stats.json',train.stats)
  dl,sampler,dlgen=loader(train,micro);model=Policy(mode).cuda();atomic_json(directory/'parameter_audit.json',model.audit())
  opt=MasterAdamW(model);generator=torch.Generator(device='cuda').manual_seed(42);step=0;best=float('inf')
  if resume:
   state=torch.load(directory/'recovery.pt',map_location='cpu',weights_only=False)
   assert state['config']==config,'Resume configuration mismatch'
   model.load_trainables(state['weights']);opt.load_state_dict(state['optimizer']);step=state['step'];best=state['best']
   restore_random(state['random'],generator,sampler,dlgen)
   artifact=dict(weights=state['weights'],step=step,config=config,best=best)
   save_torch(directory/'last.pt',artifact)
   if state.get('is_best'):save_torch(directory/'best.pt',artifact)
   del artifact,state
  # DataLoader iterator construction advances only its dedicated generator. Preserve it on resume.
  saved_loader=dlgen.get_state();iterator=iter(dl)
  if resume:dlgen.set_state(saved_loader)
  started=time.time();model.train();torch.cuda.reset_peak_memory_stats()
  atomic_json(status,dict(state='running',step=step,pid=os.getpid(),task=task,mode=mode,started=started))
  try:
   while step<config['steps']:
    disk_check();t0=time.time();report=update(model,opt,iterator,generator,step+1,micro,audit=step<2);step+=1
    row=dict(step=step,loss=report['loss'],grad_norm=report['grad_norm'],seconds=time.time()-t0,lrs=[g['lr'] for g in opt.optimizer.param_groups],peak_cuda_bytes=torch.cuda.max_memory_allocated())
    if step%config['validation_interval']==0 or step==config['steps']:
     result=validate(model,val,micro);row['validation']=result;is_best=result['loss']<best;best=min(best,result['loss'])
     checkpoint(directory,model,opt,step,best,generator,sampler,dlgen,config,is_best)
    elif step%100==0:checkpoint(directory,model,opt,step,best,generator,sampler,dlgen,config)
    with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    atomic_json(status,dict(state='running',step=step,total=config['steps'],pid=os.getpid(),elapsed=time.time()-started,**{k:v for k,v in row.items() if k!='step'}))
    print(json.dumps(row),flush=True)
   atomic_json(status,dict(state='completed',step=step,total=config['steps'],best_validation=best,elapsed=time.time()-started))
   # Only this run's disposable optimizer recovery state is removed after verified final save.
   (directory/'recovery.pt').unlink()
  except BaseException:
   atomic_json(status,dict(state='failed',step=step,error=traceback.format_exc(),pid=os.getpid()))
   raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.task,a.mode,a.output,a.resume)
if __name__=='__main__':main()
