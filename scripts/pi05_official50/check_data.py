from common import TASKS,OUT,atomic_json
from data import Episodes,targets,normalize,unnormalize,stats_for
import numpy as np
rows=[]
for task in TASKS:
 s=stats_for(task)
 for mode in ['rgb','rgb_tactile']:
  d=Episodes(task,mode);i,t=d.index[0];x=targets(d.q[i],d.a[i],t);y=unnormalize(normalize(x,s['action']),s['action']);y[:,:7]+=d.q[i][t,:7];np.testing.assert_allclose(y,d.a[i][t:t+50],rtol=1e-5,atol=1e-6)
  first=d[0];last=d[len(d)-1];assert int(last['valid'].sum())==8;assert not last['actions'][1:].any();assert not first['valid'][:,8:].any()
  row=dict(task=task,mode=mode,samples=len(d),passed=True);rows.append(row);print(row,flush=True)
atomic_json(OUT/'setup/data_audit.json',rows)
