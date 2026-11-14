"""Persistent five-slot queue; a failed job prevents subsequent launches."""
import argparse,fcntl,json,os,subprocess,time
from common import OUT,TASKS,MODES,ROOT,atomic_json,RECIPE

def schedule(jobs,launch,publish,limit=5,interval=2):
 pending=list(jobs);active={};completed=[];failed={}
 while pending or active:
  # Collect every exit before refilling slots, so an error cannot be missed.
  for name,(process,log) in list(active.items()):
   code=process.poll()
   if code is None:continue
   log.close();del active[name]
   if code==0:completed.append(name)
   else:failed[name]=code
  while pending and len(active)<limit and not failed:
   name=pending.pop(0)
   try:active[name]=launch(name)
   except Exception as exc:failed[name]=repr(exc)
  publish(dict(state='draining_after_error' if failed and active else 'stopped_on_error' if failed else 'running' if active else 'completed',active=list(active),pids={n:p.pid for n,(p,_) in active.items()},queued=pending,completed=completed,failed=failed,concurrency=limit))
  if failed and not active:break
  if active:time.sleep(interval)
 return 1 if failed else 0

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');a=p.parse_args()
 OUT.mkdir(exist_ok=True,parents=True)
 with (OUT/'queue.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  acceptance=json.loads((OUT/'acceptance.json').read_text());assert acceptance['passed'] and acceptance['concurrency']==5
  jobs=[f'{t}_{m}' for t in TASKS[:2] for m in MODES]+['pull_out_key_rgb_tactile','pull_out_key_rgb']
  for name in jobs:
   status=OUT/'runs'/name/'status.json'
   if status.exists():raise RuntimeError(f'{name} has existing state; this campaign requires fresh outputs')
  def launch(name):
   directory=OUT/'runs'/name;directory.mkdir(parents=True,exist_ok=True)
   atomic_json(directory/'status.json',dict(state='initializing',step=0,total=RECIPE['steps']))
   log=(directory/'train.log').open('w')
   try:process=subprocess.Popen([str(ROOT/'scripts/pi05_official50'/f'train_{name}.sh'),'--gpu',a.gpu],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   except BaseException:log.close();raise
   return process,log
  def publish(state):atomic_json(OUT/'queue.json',dict(state,pid=os.getpid(),gpu=a.gpu))
  raise SystemExit(schedule(jobs,launch,publish))
if __name__=='__main__':main()
