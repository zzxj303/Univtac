"""Seal completed checks and storage estimate before any production job starts."""
import json,shutil,subprocess
from pathlib import Path
import torch
from common import *
assert json.loads((OUT/'setup/weighting_audit.json').read_text())['passed']
reports=[json.loads((OUT/'acceptance'/mode/'report.json').read_text()) for mode in MODES]
assert all(r['passed'] and r['optimizer_steps']>=20 and r['effective_batch']==32 for r in reports)
assert all(r['micro_batch']==RECIPE['micro_batch']==8 for r in reports)
assert RECIPE['accumulation']==4
assert len({r['initial_lora_hash'] for r in reports})==1
assert all(r['passed'] for r in json.loads((OUT/'setup/data_audit.json').read_text()))
manifest=json.loads((OUT/'base/manifest.json').read_text());assert sha256(BASE)==manifest['sha256']
rows=json.loads((OUT/'acceptance/rgb/parameters.json').read_text())
train_bytes=sum(r['numel']*(2 if r['dtype']=='torch.bfloat16' else 4) for r in rows if r['trainable'])
master_bytes=sum(r['numel']*4 for r in rows if r['trainable'] and r['dtype']=='torch.bfloat16')
optimizer_bytes=2*sum(r['numel']*4 for r in rows if r['trainable'])+master_bytes
# Six best+last pairs, five active recovery files, one serialized replacement, 1 GiB overhead.
required=12*train_bytes+6*(train_bytes+optimizer_bytes)+1024**3
free=shutil.disk_usage(OUT).free
assert free-required>=8*1024**3,(free,required)
for task in TASKS:
 for mode in MODES:
  script=Path(__file__).parent/f'train_{task}_{mode}.sh'
  assert f'launch.sh\" {task} {mode} ' in script.read_text(),script
for script in Path(__file__).parent.glob('*.sh'):subprocess.run(['bash','-n',str(script)],check=True)
capacity=torch.cuda.get_device_properties(0).total_memory
assert 5*max(r['peak_reserved_bytes'] for r in reports)+3*1024**3<capacity
assert json.loads((OUT/'setup/schedule_4000_audit.json').read_text())['passed']
report=dict(passed=True,training_recipe=RECIPE,validation_reuse="Model/data/micro8 acceptance retained; 4000-step schedule and optimizer resume checked separately",micro_batch=reports[0]['micro_batch'],concurrency=5,concurrency_reason='User approved five concurrent jobs; pull_out_key_rgb refills a free slot',base_sha256=manifest['sha256'],modalities=reports,storage=dict(free_bytes=free,worst_case_additional_bytes=required,reserve_bytes=8*1024**3,trainable_checkpoint_bytes=train_bytes,optimizer_bytes=optimizer_bytes),hardware=versions(),script_hashes={p.name:sha256(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.sh')})
atomic_json(OUT/'acceptance.json',report)
atomic_json(OUT/'memory_report.json',dict(gpu_bytes=capacity,modalities=reports,storage=report['storage'],concurrency=5))
print(json.dumps({'passed':True,'micro_batch':report['micro_batch'],'storage':report['storage']},indent=2))
