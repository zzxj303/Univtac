"""Adopt live training jobs and add the user-requested fifth concurrent run."""
import fcntl,json,os,subprocess,time
from pathlib import Path
from common import OUT,ROOT,atomic_json,RECIPE

def identity(pid):
 p=Path(f'/proc/{pid}')
 try:
  args=(p/'cmdline').read_bytes().decode().split('\0')
  if not any(a.endswith('/pi05_official50/train.py') for a in args):return None
  return (p/'stat').read_text().rsplit(')',1)[1].split()[19]
 except (OSError,IndexError):return None

class Attached:
 def __init__(self,name,pid):
  self.name=name;self.pid=pid;self.start=identity(pid)
  assert self.start is not None,(name,pid)
 def poll(self):
  if identity(self.pid)==self.start:return None
  p=OUT/'runs'/self.name/'status.json'
  return 0 if p.exists() and json.loads(p.read_text()).get('state')=='completed' else 1

def main():
 with (OUT/'queue.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=json.loads((OUT/'setup/five_slot_plan.json').read_text())
  active={name:(Attached(name,pid),None) for name,pid in plan['adopt_pids'].items()}
  pending=list(plan['pending']);completed=[];failed={}
  while active or pending:
   for name,(p,log) in list(active.items()):
    rc=p.poll()
    if rc is None:continue
    if log:log.close()
    del active[name]
    if rc==0:completed.append(name)
    else:failed[name]=rc
   while pending and len(active)<5 and not failed:
    name=pending.pop(0);directory=OUT/'runs'/name;log=None
    try:
     assert not (directory/'status.json').exists(),f'Existing run: {name}'
     directory.mkdir(parents=True,exist_ok=True)
     atomic_json(directory/'status.json',dict(state='initializing',step=0,total=RECIPE['steps']))
     log=(directory/'train.log').open('w')
     p=subprocess.Popen([str(ROOT/'scripts/pi05_official50'/f'train_{name}.sh'),'--gpu',plan['gpu']],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
     active[name]=(p,log)
    except Exception as exc:
     if log:log.close()
     failed[name]=repr(exc)
   state='draining_after_error' if failed and active else 'stopped_on_error' if failed else 'running' if active else 'completed'
   atomic_json(OUT/'queue.json',dict(state=state,pid=os.getpid(),gpu=plan['gpu'],concurrency=5,active=list(active),pids={n:p.pid for n,(p,_) in active.items()},queued=pending,completed=completed,failed=failed,controller='campaign_five.py'))
   if failed and not active:break
   if active:time.sleep(2)
  raise SystemExit(1 if failed else 0)
if __name__=='__main__':main()
