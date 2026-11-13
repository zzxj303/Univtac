"""Real GPU offline acceptance. Never constructs a simulator or uses evaluation seeds."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse,gc,hashlib,json,time
import torch,numpy as np
from torch.utils.data import default_collate,Subset
from common import *
from data import Episodes,loader,to_device
from model import Policy,MasterAdamW
from train import update,flow_random,random_state,restore_random,validate,save_torch

def frozen_hash(model):
 h=hashlib.sha256()
 for n,p in model.named_parameters():
  if not p.requires_grad:h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()

def category(n):
 if not n.startswith('paligemma_with_expert.paligemma.'):return 'expert'
 if 'multi_modal_projector' in n:return 'vision_projector'
 if 'vision_tower' in n:return 'vision_attention' if 'self_attn' in n else 'vision_mlp'
 return 'language_attention' if 'self_attn' in n else 'language_mlp'

def tensor_hash(values):
 h=hashlib.sha256()
 for n,p in values.items():h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()

@torch.no_grad()
def prediction(model,batch):
 obs,actions,valid=to_device(batch);g=torch.Generator(device='cuda').manual_seed(42);noise,_=flow_random(actions,g)
 model.eval()
 with torch.autocast('cuda',dtype=torch.bfloat16):a=model.sample_actions('cuda',obs,noise=noise,num_steps=10)
 assert torch.isfinite(a).all();model.train();return a.detach().cpu()

def run(mode,micro):
 seed_all();directory=OUT/'acceptance'/mode;directory.mkdir(parents=True,exist_ok=True)
 model=Policy(mode).cuda();rows=model.audit();atomic_json(directory/'parameters.json',rows)
 initial={n:p.detach().cpu().clone() for n,p in model.trainables().items()};before=frozen_hash(model)
 initial_lora=tensor_hash({n:p for n,p in initial.items() if n.endswith(('.A','.B'))})
 # A second complete seed-42 initialization must produce identical adapter tensors.
 seed_all();second=Policy(mode);assert tensor_hash({n:p for n,p in second.trainables().items() if n.endswith(('.A','.B'))})==initial_lora
 del second;gc.collect()
 dataset=Episodes('insert_HDMI',mode);dl,sampler,dlgen=loader(dataset,micro);iterator=iter(dl)
 dl2,sampler2,gen2=loader(dataset,micro);iterator2=iter(dl2)
 first=next(iterator);other=next(iterator2);assert torch.equal(first['index'],other['index'])
 # Reset both sampler and loader before the measured optimization sequence.
 dl,sampler,dlgen=loader(dataset,micro);iterator=iter(dl)
 generator=torch.Generator(device='cuda').manual_seed(42);g2=torch.Generator(device='cuda').manual_seed(42)
 dummy=torch.zeros((micro,50,32),device='cuda');n1,t1=flow_random(dummy,generator);n2,t2=flow_random(dummy,g2);assert torch.equal(n1,n2) and torch.equal(t1,t2);generator.manual_seed(42)
 opt=MasterAdamW(model);model.train();torch.cuda.reset_peak_memory_stats();records=[];started=time.time()
 for step in range(1,21):
  t=time.time();result=update(model,opt,iterator,generator,step,micro,audit=step in (1,2,20));torch.cuda.synchronize()
  records.append({'step':step,'loss':result['loss'],'grad_norm':result['grad_norm'],'seconds':time.time()-t,'peak_bytes':torch.cuda.max_memory_allocated()})
  if result['gradients']:atomic_json(directory/f'gradients_{step}.json',result['gradients'])
  atomic_json(directory/'progress.json',records);print(mode,records[-1],flush=True)
 assert frozen_hash(model)==before,'Frozen weights changed'
 for state in opt.optimizer.state.values():
  for value in state.values():
   if torch.is_tensor(value) and value.is_floating_point():assert value.dtype==torch.float32,value.dtype
 changed={};gradcategories={}
 for n,p in model.trainables().items():
  c=category(n);changed[c]=changed.get(c,False) or not torch.equal(initial[n],p.detach().cpu())
  gradcategories[c]=gradcategories.get(c,0)+result['gradients'][n]
 assert all(changed.values()) and all(v>0 for v in gradcategories.values()),(changed,gradcategories)
 del initial;gc.collect()
 # Verify each additional tactile view reaches visual features (and masks are true).
 obs,_,_=to_device(first)
 tactile={}
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for key in keys(mode):
   feature=model.paligemma_with_expert.embed_image(obs.images[key]);altered=model.paligemma_with_expert.embed_image(torch.zeros_like(obs.images[key]))
   assert not torch.equal(feature,altered);tactile[key]=float((feature-altered).float().norm())
 offline_tasks={}
 for task in TASKS:
  sample=default_collate([Episodes(task,mode)[0]])
  output=prediction(model,sample);offline_tasks[task]={'shape':list(output.shape),'finite':bool(torch.isfinite(output).all())}
 predicted=prediction(model,first)
 weights={n:p.detach().cpu().clone() for n,p in model.trainables().items()}
 save_torch(directory/'reload.pt',weights);model.load_trainables(torch.load(directory/'reload.pt',weights_only=True));assert torch.equal(predicted,prediction(model,first));(directory/'reload.pt').unlink()
 # Validation leaves every training/global random stream unchanged.
 rng=random_state(generator,sampler,dlgen);before_rng=tensor_hash({'noise':rng['noise'],'torch':rng['torch'],'cuda':rng['cuda'][0]})
 validation=validate(model,Subset(Episodes('insert_HDMI',mode,'validation_ids'),range(8)),micro)
 after_rng=random_state(generator,sampler,dlgen);assert before_rng==tensor_hash({'noise':after_rng['noise'],'torch':after_rng['torch'],'cuda':after_rng['cuda'][0]})
 assert rng['python']==after_rng['python'] and np.array_equal(rng['numpy'][1],after_rng['numpy'][1])
 # Persist and restore all masters/optimizer moments and replay the next full batch.
 save_torch(directory/'resume.pt',{'weights':weights,'optimizer':opt.state_dict(),'random':random_state(generator,sampler,dlgen),'step':20})
 r1=update(model,opt,iterator,generator,21,micro);state_hash=tensor_hash(model.trainables());master_hash=tensor_hash(opt.masters)
 restored=torch.load(directory/'resume.pt',map_location='cpu',weights_only=False)
 model.load_trainables(restored['weights']);opt.load_state_dict(restored['optimizer']);restore_random(restored['random'],generator,sampler,dlgen)
 saved=dlgen.get_state();iterator=iter(dl);dlgen.set_state(saved)
 r2=update(model,opt,iterator,generator,restored['step']+1,micro)
 assert r1['indices']==r2['indices'] and r1['loss']==r2['loss'],(r1['loss'],r2['loss'])
 assert tensor_hash(model.trainables())==state_hash and tensor_hash(opt.masters)==master_hash,'Resume mismatch'
 (directory/'resume.pt').unlink()
 report={'benchmark_context':os.environ.get('PI05_ACCEPTANCE_CONTEXT','single process'),'passed':True,'mode':mode,'micro_batch':micro,'optimizer_steps':20,'effective_batch':32,'seconds':time.time()-started,'mean_update_seconds':sum(r['seconds'] for r in records[2:])/18,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'changed_categories':changed,'gradient_categories':gradcategories,'view_effect_norm':tactile,'offline_tasks':offline_tasks,'validation_subset':validation,'frozen_hash':before,'initial_lora_hash':initial_lora,'resume_exact':True,'reload_exact':True,'seed42_reproducible':True,'validation_rng_independent':True}
 atomic_json(directory/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--micro',type=int,default=RECIPE['micro_batch']);a=p.parse_args();assert 32%a.micro==0;run(a.mode,a.micro)
