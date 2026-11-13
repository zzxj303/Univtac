"""Compare full effective-batch-32 updates at micro 4/8/16/32, never run production jobs."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse,gc,statistics,time
import torch
from common import OUT,MODES,seed_all,atomic_json,versions
from data import Episodes,loader
from model import Policy,MasterAdamW
from train import update

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=MODES,required=True);a=p.parse_args()
 seed_all();directory=OUT/'micro_batch_benchmark';directory.mkdir(parents=True,exist_ok=True)
 model=Policy(a.mode).cuda();model.audit();initial={n:v.detach().cpu().clone() for n,v in model.trainables().items()}
 dataset=Episodes('insert_HDMI',a.mode);rows=[];model.train()
 for micro in (4,8,16,32):
  print('START',a.mode,'micro',micro,flush=True)
  model.load_trainables(initial);seed_all();opt=MasterAdamW(model)
  dl,sampler,dlgen=loader(dataset,micro);iterator=iter(dl);generator=torch.Generator(device='cuda').manual_seed(42)
  torch.cuda.reset_peak_memory_stats();records=[]
  try:
   for step in range(1,8):
    torch.cuda.synchronize();start=time.perf_counter()
    r=update(model,opt,iterator,generator,step,micro,audit=step==1)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    records.append({'step':step,'seconds':elapsed,'loss':r['loss'],'grad_norm':r['grad_norm']})
    print(a.mode,micro,records[-1],flush=True)
   times=[r['seconds'] for r in records[2:]]
   row={'mode':a.mode,'micro_batch':micro,'accumulation':32//micro,'effective_batch':32,'warmup_updates':2,'measured_updates':5,'mean_update_seconds':statistics.mean(times),'median_update_seconds':statistics.median(times),'samples_per_second':32/statistics.mean(times),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'finite_loss_and_gradients':True,'gradient_connections_checked':True,'records':records}
  except torch.cuda.OutOfMemoryError as exc:
   row={'mode':a.mode,'micro_batch':micro,'oom':True,'error':str(exc),'records':records}
  rows.append(row);atomic_json(directory/f'{a.mode}.json',rows)
  print('RESULT',row,flush=True)
  opt.zero_grad();del opt,iterator,dl,generator;gc.collect();torch.cuda.empty_cache()
  if row.get('oom'):break
 atomic_json(directory/f'{a.mode}_environment.json',versions())
if __name__=='__main__':main()
