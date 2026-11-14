import json
from common import OUT,TASKS,MODES,RECIPE
if (OUT/'queue.json').exists():print((OUT/'queue.json').read_text())
for t in TASKS:
 for m in MODES:
  p=OUT/'runs'/f'{t}_{m}'/'status.json'
  s=json.loads(p.read_text()) if p.exists() else {'state':'queued','step':0}
  print(f"{t:14} {m:12} {s['state']:18} {s.get('step',0):5}/{s.get('total',RECIPE['steps'])}",'loss=',s.get('loss'),'seconds/step=',s.get('seconds'))
