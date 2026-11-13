"""Launch the twelve fixed shell entries with live concurrency and resource checks."""
import fcntl,json,os,subprocess,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/official50/evaluation_launch'
BASE.mkdir(parents=True,exist_ok=True)
lock=(BASE/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
control=BASE/'control.json'
if not control.exists():control.write_text(json.dumps({'concurrency':1,'pause':False}))
jobs=[{'case':f'{p}_{t}_{m}','state':'queued'} for t in ['insert_HDMI','insert_hole','pull_out_key'] for p in ['act','dp'] for m in ['rgb','rgb_tactile']]
active={};blocked=False

def status():
 tmp=BASE/'status.tmp';tmp.write_text(json.dumps({'pid':os.getpid(),'updated':time.time(),'blocked':blocked,'jobs':jobs},indent=2));tmp.replace(BASE/'status.json')

while True:
 for i,(process,log) in list(active.items()):
  code=process.poll()
  if code is not None:
   log.close();jobs[i].update(state='completed' if code==0 else 'halted',returncode=code,ended=time.time());del active[i]
   if code:blocked=True
 cfg=json.loads(control.read_text())
 if not blocked and not cfg['pause'] and len(active)<cfg['concurrency']:
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
  if free>=16000 and shutil.disk_usage(BASE).free>=8*2**30:
   queued=next((i for i,j in enumerate(jobs) if j['state']=='queued'),None)
   if queued is not None:
    job=jobs[queued];log=(BASE/(job['case']+'.log')).open('ab',buffering=0)
    command=['bash',str(ROOT/'scripts/official50/evaluation'/(job['case']+'.sh')),'--gpu','0']
    process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    job.update(state='running',pid=process.pid,started=time.time());active[queued]=(process,log)
    print('START',job['case'],process.pid,flush=True)
 status()
 if not active and (blocked or not any(j['state']=='queued' for j in jobs)):break
 time.sleep(15)
