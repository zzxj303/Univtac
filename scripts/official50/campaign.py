"""Durable configurable queue with adoption of existing training processes. Each config is launched once; failures are recorded."""
import argparse,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/official50';PYTHON=ROOT/'.venv/bin/python'
p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
jobs=[]
for task in ['insert_HDMI','insert_hole','pull_out_key']:
 for mode in ['rgb','rgb_tactile']:
  for policy in ['act','dp']:
   out=BASE/policy/task/mode;out.mkdir(parents=True,exist_ok=True)
   jobs.append({'policy':policy,'task':task,'modality':mode,'output':str(out),'state':'completed' if (out/'completed.json').exists() else 'queued'})
class AdoptedProcess:
 def __init__(self,pid,output): self.pid=pid;self.output=Path(output)
 def poll(self):
  try:
   state=Path(f'/proc/{self.pid}/stat').read_text().split(') ',1)[1].split()[0]
   if state!='Z':return None
  except FileNotFoundError:pass
  return 0 if (self.output/'completed.json').exists() else 1
active={};started=time.time()
previous=BASE/'status.json'
if previous.exists():
 old=json.loads(previous.read_text());started=old['started']
 for i,job in enumerate(old['jobs']):
  if job['state']=='running':
   pid=job['pid']
   try: cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00',' ')
   except FileNotFoundError: cmd=''
   if f"run_{job['policy']}.py" in cmd and job['task'] in cmd:
    active[i]=(AdoptedProcess(pid,job['output']),open(os.devnull,'w'))
   else:job['state']='completed' if (Path(job['output'])/'completed.json').exists() else 'failed'
  jobs[i]=job
def status():
 tmp=BASE/'status.tmp';tmp.write_text(json.dumps({'scheduler_pid':os.getpid(),'started':started,'updated':time.time(),'workers':a.workers,'jobs':jobs},indent=2));tmp.replace(BASE/'status.json')
while True:
 for i,(proc,log) in list(active.items()):
  result=proc.poll()
  if result is not None:
   log.close();job=jobs[i];job.update(returncode=result,ended=time.time(),state='completed' if result==0 and (Path(job['output'])/'completed.json').exists() else 'failed');del active[i]
 for i,job in enumerate(jobs):
  if len(active)>=a.workers:break
  if job['state']!='queued':continue
  if __import__('shutil').disk_usage(BASE).free < 8*2**30:
   job['state']='blocked_disk';continue
  out=Path(job['output']);log=open(out/'train.log','ab',buffering=0)
  cmd=[str(PYTHON),'-u',str(ROOT/'scripts/official50'/f"run_{job['policy']}.py"),job['task'],job['modality']]
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
  job.update(state='running',pid=proc.pid,started=time.time(),command=cmd);active[i]=(proc,log)
  print('START',job['policy'],job['task'],job['modality'],proc.pid,flush=True)
  time.sleep(3)
 status()
 if not active and not any(j['state']=='queued' for j in jobs):break
 time.sleep(10)
print('QUEUE_FINISHED',flush=True)
