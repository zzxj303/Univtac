import json
from eval_common import ROOT,CASES,summarize
out=ROOT/'outputs/pi05_official50'
p=out/'evaluation_launch/status.json'
print(p.read_text() if p.exists() else 'Campaign not started')
for case in CASES:
    s=summarize(out/'evaluation'/case,case)
    print(case, f"{s['recorded']}/30",s['counts'],'pending=',s['pending'])
