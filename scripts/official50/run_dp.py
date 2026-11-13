import os,sys,json,time,random,atexit
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np,torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from dp_adapter import create
import dp_train_loop as loop

def main():
 task,mode=sys.argv[1:3];smoke='--smoke' in sys.argv
 out=ROOT/'outputs/official50'/('smoke_dp' if smoke else 'dp')/task/mode;out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
 nets,loaders,names,stats,split=create(task,mode)
 config={'task':task,'modality':mode,'encoder':'resnet18','pretrained':False,'camera_names':names,'state_dim':8,'action_dim':8,'pred_horizon':20,'obs_horizon':1,'epochs':3500,'batch_size':8,'validation_batch_size':2,'lr':1e-4,'weight_decay':1e-6,'warmup_steps':500,'schedule':'cosine','diffusion_timesteps':100,'seed':42,'train_episodes':40,'validation_episodes':10,'normalization_scope':'all 50 episodes, matching upstream','tactile_preprocess':'rgb_marker uint8 / 255','upstream':'policy/ViTAL/diffusion/train_policy.py','ema':False}
 (out/'config.json').write_text(json.dumps(config,indent=2));(out/'split.json').write_text(json.dumps(split,indent=2));np.savez(out/'dataset_stats.npz',**stats)
 start=time.time();best=float('inf');last_epoch=-1
 def save(path,epoch):
  tmp=path.with_suffix('.tmp');torch.save({'state_dict':nets.state_dict(),'epoch':epoch,'config':config,'stats':stats},tmp);tmp.replace(path)
 def record(epoch,train,val,model,validated):
  nonlocal best,last_epoch
  last_epoch=epoch
  row={'epoch':epoch,'optimizer_steps':(epoch+1)*len(loaders[0]),'train_loss':train,'seconds':time.time()-start}
  if validated:
   if not np.isfinite(val):raise RuntimeError('Nonfinite validation loss')
   row['validation_loss']=val
   if val<best:
    best=val;save(out/'policy_best.ckpt',epoch)
  with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if epoch%10==0:print('METRICS',row,flush=True)
 def periodic(start_time,epoch,enc_type,nets,train_losses,val_losses):
  save(out/'policy_periodic.ckpt',epoch)
  if epoch in [1,3500]:save(out/'policy_last.ckpt',last_epoch)
 def memory():
  d={'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'seconds':time.time()-start}
  (out/'memory.json').write_text(json.dumps(d,indent=2));print('MEMORY',d,flush=True)
 atexit.register(memory)
 loop.CKPT_DIR=str(out)+'/'
 loop.record_epoch=record;loop._save_ckpt=periodic
 loop.noise_scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2',clip_sample=True,prediction_type='epsilon')
 loop.train(1 if smoke else 3500,names,nets,*loaders,'resnet18')
 (out/'completed.json').write_text(json.dumps({'completed':True,'epochs':1 if smoke else 3500,'best_validation_loss':best,'seconds':time.time()-start}))
if __name__=='__main__':main()
