"""Bounded copy/replay/repair transactions. Every attempted simulation is charged."""
import json,time,uuid
from pathlib import Path
import h5py,numpy as np
from orchestrator.common import ROOT,PYTHON,atomic,read,writable,lock,environment,run,audit,gpu_slot
from adapters.datasets import catalog,physical_copy,convert,create_version

def session_path(path):
    p=writable(path)
    if not p.is_relative_to(ROOT/'runs/repairs'):raise ValueError('Unregistered repair session')
    return p

def patch_commands(source,dest,patch):
    physical_copy(source,dest)
    start,stop=patch['start'],patch['stop']
    with h5py.File(dest,'r+') as f:
        data=f['action'];n=len(data)
        if type(start)!=int or type(stop)!=int or not 0<=start<stop<=n:raise ValueError('Invalid repair window')
        if patch['mode']=='offset':
            offset=np.asarray(patch['values'],dtype=np.float32)
            if offset.shape!=(8,):raise ValueError('Offset requires eight numbers')
            # A tapered window has unchanged boundaries, avoiding discontinuous joins.
            weights=np.sin(np.linspace(0,np.pi,stop-start))[:,None] if stop-start>2 else np.ones((stop-start,1))
            value=data[start:stop]+weights*offset
        elif patch['mode']=='replace':
            value=np.asarray(patch['values'],dtype=np.float32)
            if value.shape!=(stop-start,8):raise ValueError('Replacement shape mismatch')
        else:raise ValueError('Unknown patch mode')
        if not np.isfinite(value).all():raise ValueError('Nonfinite repair')
        # Bound command changes relative to a single joint controller update. Native limits are checked again in simulation.
        original=data[:];candidate=original.copy();candidate[start:stop]=value
        limits=np.maximum(np.max(np.abs(np.diff(original,axis=0)),axis=0)*2,np.array([.05]*7+[.005]))
        if np.any(np.abs(np.diff(candidate,axis=0))>limits+1e-6):raise ValueError('Repair introduces an excessive command discontinuity')
        data[start:stop]=value
    return str(dest)

def _replay(session,demo,kind,commands):
    folder=session_path(session);m=read(folder/'session.json');entry=m['demos'][str(demo)]
    trials=[r for r in m['trials'] if r['demo_id']==demo]
    existing=next((r for r in trials if r['kind']==kind),None)
    if existing:
        result=Path(existing['folder'])/'result.json'
        if not result.exists():raise RuntimeError('Interrupted simulation requires reconciliation; no hidden retry')
        return read(result)
    if len(trials)>=4:raise ValueError('Four-trial demonstration budget exhausted')
    out=folder/str(demo)/kind;out.mkdir(parents=True,exist_ok=True)
    row={'demo_id':demo,'kind':kind,'folder':str(out),'started_at':time.time()}
    m['trials'].append(row);atomic(folder/'session.json',m)
    request={'task':m['group']['task'],'seed':entry['source_seed'],'commands':str(commands),'reference':entry['reference'],'folder':str(out)}
    atomic(out/'request.json',request,exclusive=True)
    with gpu_slot():
        result=run([str(PYTHON),'-m','adapters.replay_worker','--request',str(out/'request.json'),'--headless'],out/'process',
                   env=environment({'TH_OMNI_LOG_DIR':out/'extension-logs','LD_PRELOAD':ROOT/'runtime/io_probe.so'}),timeout=1800)
    if not (out/'result.json').exists():atomic(out/'result.json',{'success':False,'error':'worker terminated','process':result,'phase':'infrastructure'})
    audit();return read(out/'result.json')

def prepare(group,intents,folder,parent=None):
    folder=session_path(folder);folder.mkdir(parents=True,exist_ok=True)
    rows={r['demo_id']:r for r in catalog(group)}
    ids=[r['demo_id'] for r in intents]
    if not 1<=len(ids)<=3 or len(set(ids))!=len(ids):raise ValueError('Select 1..3 distinct demonstrations')
    with lock(folder/'.lock'):
        if (folder/'session.json').exists():
            m=read(folder/'session.json')
            if m['group']!=group or m['intents']!=intents or m['parent']!=parent:raise ValueError('Repair session identity changed')
        else:
            m={'group':group,'intents':intents,'parent':parent,'demos':{},'trials':[],'created_at':time.time()}
            for intent in intents:
                i=intent['demo_id'];r=rows[i]
                if not r['eligible_for_repair']:raise ValueError('Demo/seed is not eligible for repair')
                d=folder/str(i);d.mkdir()
                copied=physical_copy(r['raw_path'],d/'source.hdf5')
                if r['training_path']:physical_copy(r['training_path'],d/'source_commands.hdf5')
                else:convert(d/'source.hdf5',d/'source_commands.hdf5')
                if parent:
                    from adapters.datasets import validate_version
                    previous=validate_version(parent,group)
                    if str(i) in previous['episodes'] and previous['episodes'][str(i)].get('modified'):
                        # Keep source copy immutable; current-version commands form a separate base.
                        physical_copy(previous['episodes'][str(i)]['path'],d/'current_commands.hdf5')
                base=d/'current_commands.hdf5' if (d/'current_commands.hdf5').exists() else d/'source_commands.hdf5'
                m['demos'][str(i)]={**r,'base_commands':str(base),'reference':str(base),'copy':copied,'intent':intent}
            atomic(folder/'session.json',m,exclusive=True)
        for i in ids:_replay(folder,i,'original',m['demos'][str(i)]['base_commands'])
    return str(folder)

def try_repair(session,demo_id,patch):
    folder=session_path(session)
    with lock(folder/'.lock'):
        m=read(folder/'session.json');entry=m['demos'][str(demo_id)]
        original=read(folder/str(demo_id)/'original/result.json')
        if original.get('error') and original.get('phase') != 'validate_commands':raise ValueError('Original replay infrastructure/control failure must be resolved first')
        attempts=[r for r in m['trials'] if r['demo_id']==demo_id and r['kind'].startswith('candidate-')]
        if len(attempts)>=2:raise ValueError('Two repair candidates already attempted')
        if any(r['demo_id']==demo_id and r['kind']=='verify' for r in m['trials']):raise ValueError('Verification already attempted')
        intent=entry['intent']
        if patch['start']<intent['start'] or patch['stop']>intent['stop']:raise ValueError('Patch exceeds declared problem window')
        kind=f'candidate-{len(attempts)+1}';out=folder/str(demo_id)/kind
        proposal=folder/str(demo_id)/('proposal-'+uuid.uuid4().hex);proposal.mkdir()
        atomic(proposal/'patch.json',patch,exclusive=True)
        patch_commands(entry['base_commands'],proposal/'commands.hdf5',patch)
        proposal.rename(out)
        commands=str(out/'commands.hdf5')
        result=_replay(folder,demo_id,kind,commands)
        return {'candidate':kind,'result':result,'source':str(out/'result.json')}

def verify_repair(session,demo_id,candidate):
    folder=session_path(session)
    if candidate not in ('candidate-1','candidate-2'):raise ValueError('Unknown candidate')
    with lock(folder/'.lock'):
        selected=folder/str(demo_id)/candidate
        trial=read(selected/'result.json');original=read(folder/str(demo_id)/'original/result.json')
        if not trial.get('success'):raise ValueError('Candidate must succeed before independent verification')
        proof=folder/str(demo_id)/'acceptance.json'
        if proof.exists():
            old=read(proof)
            if old['candidate']!=candidate:raise ValueError('Cannot replace sealed acceptance')
            return old
        verification=_replay(folder,demo_id,'verify',selected/'commands.hdf5')
        improved=not original.get('success') or (trial.get('tracking_rmse',float('inf'))<original['tracking_rmse'] and (verification.get('tracking_rmse') if verification.get('tracking_rmse') is not None else float('inf'))<original['tracking_rmse'])
        accepted=bool(trial.get('success') and verification.get('success') and improved)
        m=read(folder/'session.json')
        result={'accepted':accepted,'demo_id':demo_id,'group':m['group']['id'],'task':m['group']['task'],'scope':read(folder/'fixture.json')['scope'] if (folder/'fixture.json').exists() else 'formal demonstration repair','candidate':candidate,'recording':verification.get('recording'),
                'criterion':'native success twice; if original successful, lower reference tracking RMSE twice','candidate_result':str(selected/'result.json'),
                'verification_result':str(folder/str(demo_id)/'verify/result.json'),'acceptance_file':str(proof)}
        atomic(proof,result,exclusive=True);return result

def accepted_replacements(session):
    folder=session_path(session)
    return {int(p.parent.name):str(p) for p in folder.glob('*/acceptance.json') if read(p)['accepted']}

def repair_sources(session):
    """Allow only registered demo artifacts, never CLI homes, config or credentials."""
    root=session_path(session);m=read(root/'session.json')
    result=[str(root/'session.json')]
    if (root/'fixture.json').is_file():result.append(str(root/'fixture.json'))
    names={'source.hdf5','reference.hdf5','source_commands.hdf5','current_commands.hdf5','commands.hdf5','recording.hdf5',
           'result.json','patch.json','acceptance.json','trajectory.jsonl','protocol.json','progress.json','metadata.json'}
    for demo in m['demos']:
        if not demo.isdigit():raise ValueError('Invalid registered demonstration ID')
        folder=root/demo
        for p in folder.rglob('*'):
            if p.is_file() and p.name in names and p.resolve().is_relative_to(folder) and not any(x.startswith(('kit','process','extension')) for x in p.relative_to(folder).parts):
                result.append(str(p.resolve()))
    return sorted(set(result))
