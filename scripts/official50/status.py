import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'outputs/official50/status.json'
if not p.exists():raise SystemExit('Campaign has not started')
s=json.loads(p.read_text())
for j in s['jobs']:
 out=Path(j['output']);detail=''
 metrics=out/'metrics.jsonl'
 if metrics.exists():
  row=json.loads(metrics.read_text().splitlines()[-1]);detail=f"epoch {row['epoch']+1}/3500; loss {row['train_loss']:.5f}"
 elif (out/'train.log').exists():
  log=(out/'train.log').read_text(errors='replace');matches=re.findall(r'(\d+)/4000',log)
  if matches:detail=f'step {matches[-1]}/4000'
 print(f"{j['policy']:3} {j['task']:14} {j['modality']:11} {j['state']:10} {detail}")
