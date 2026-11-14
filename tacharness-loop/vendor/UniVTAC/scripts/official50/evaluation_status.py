"""Read-only progress for the twelve simulation evaluation groups."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'outputs/official50'
s=json.loads((base/'evaluation_launch/status.json').read_text())
print('Scheduler:',s['pid'],'blocked:',s['blocked'])
for job in s['jobs']:
 group=base/'evaluation'/job['case'];done=[];current=''
 if group.exists():
  for path in sorted(group.glob('[0-9]*')):
   result=path/'result.json';progress=path/'progress.json'
   if result.exists():done.append(json.loads(result.read_text()))
   elif progress.exists():
    p=json.loads(progress.read_text());current=f"seed={p['seed']} phase={p['phase']} actions={p['actions']}"
 counts={k:sum(r['status']==k for r in done) for k in ['success','failed','error','timeout','interrupted']}
 print(f"{job['case']:36} {job['state']:9} {len(done):2}/30 {counts} {current}")
