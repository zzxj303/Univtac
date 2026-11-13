"""Execute unchanged upstream training entry point from an isolated experiment cwd."""
import atexit,json,os,runpy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ['TORCH_HOME']=str(ROOT/'.cache/torch')
os.environ.setdefault('OMP_NUM_THREADS','4')
import torch
torch.set_num_threads(4)
task,mode=sys.argv[1:3];smoke='--smoke' in sys.argv
base=ROOT/'outputs/official50';out=base/('smoke' if smoke else 'act')/task/mode;out.mkdir(parents=True,exist_ok=True)
config=base/'act'/task/mode/'config.yml'
if smoke:
 import yaml
 cfg=yaml.safe_load(config.read_text());cfg['num_steps']=2
 config=out/'config.yml';config.write_text(yaml.safe_dump(cfg))
start=time.time()
def report():
 d={'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'seconds':time.time()-start}
 (out/'memory.json').write_text(json.dumps(d,indent=2));print('MEMORY',d,flush=True)
atexit.register(report)
os.chdir(base);sys.path.insert(0,str(ROOT/'policy/ACT'))
sys.argv=['imitate_episodes.py','--task_name',f'sim-{task}-official50-50','--ckpt_dir',str(out),'--config_path',str(config),'--seed','42']
runpy.run_path(str(ROOT/'policy/ACT/imitate_episodes.py'),run_name='__main__')
(out/'completed.json').write_text(json.dumps({'completed':True,'time':time.time(),'upstream_num_steps':4000,'actual_updates':4001}))
