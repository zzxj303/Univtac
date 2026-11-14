"""Fixed experiment contract; no CLI overrides for scientific settings."""
import os,random,json,hashlib,subprocess,platform
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/root/autodl-tmp/gujie/UniVTAC')
OUT=Path(os.environ['TH_OUTPUT_ROOT'])
BASE=Path('/root/autodl-tmp/gujie/UniVTAC/outputs/pi05_official50/base/model.safetensors')
TASKS=('insert_HDMI','insert_hole','pull_out_key')
MODES=('rgb','rgb_tactile')
PROMPTS={'insert_HDMI':'Insert the HDMI plug into the HDMI port.','insert_hole':'Insert the peg into the hole.','pull_out_key':'Pull the key out of the lock.'}
RECIPE=dict(seed=42,steps=4000,effective_batch=32,micro_batch=8,accumulation=4,horizon=50,action_dim=32,valid_dim=8,lora_rank=16,lora_alpha=32,lora_dropout=0,lora_lr=1e-4,expert_lr=5e-5,betas=[.9,.95],weight_decay=.01,warmup=200,min_lr_ratio=.1,grad_clip=1.,validation_interval=500,ema=False,flow_steps=10,gradient_checkpointing=True)
def seed_all():
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 torch.set_num_threads(8)
def seed_worker(worker_id):
 seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)
def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,default=str));os.replace(temp,path)
def versions():
 import transformers
 def git(p):
  return subprocess.run(['git','-C',str(p),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
 return {'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'cudnn':torch.backends.cudnn.version(),'openpi_git':git(ROOT.parent/'OpenPI-RoboLab'),'univtac_git':git(ROOT),'openpi_source_hashes':{str(p.relative_to(ROOT.parent/'OpenPI-RoboLab')):sha256(p) for p in (ROOT.parent/'OpenPI-RoboLab/src/openpi/models_pytorch').glob('*.py')},'transformers_patch_hashes':{str(p.relative_to(ROOT/'.venv-pi05')):sha256(p) for p in (ROOT/'.venv-pi05/lib/python3.11/site-packages/transformers/models').glob('*/modeling_*.py') if p.parent.name in ('gemma','paligemma','siglip')},'scripts':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},'nvidia_smi':subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout}
def keys(mode):return ['cam_high','cam_wrist']+(['tac_left','tac_right'] if mode=='rgb_tactile' else [])
