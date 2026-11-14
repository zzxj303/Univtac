from __future__ import annotations
import collections, json, math, re
from pathlib import Path
from orchestrator.common import SOURCE, ROOT, read, atomic
TARGETS = {'act_insert_hole_rgb_tactile': 'univtac_hole_act_rgbt', 'dp_insert_hole_rgb_tactile': 'univtac_hole_dp_rgbt', 'pi05_insert_hole_rgb': 'univtac_hole_pi05_rgb'}
TERMINATIONS = ('success', 'max_steps', 'early_stop', 'exception')

def discover():
    groups = []
    for family in ['official50', 'pi05_official50']:
        for folder in sorted((SOURCE / 'outputs' / family / 'evaluation').glob('*')):
            if not (folder / 'summary.json').exists():
                continue
            case = folder.name
            policy, tail = case.split('_', 1)
            mode = 'rgb_tactile' if tail.endswith('_rgb_tactile') else 'rgb'
            task = tail[:-(len(mode) + 1)]
            train = SOURCE / 'outputs' / family / ('runs' if policy == 'pi05' else policy)
            train = train / (f'{task}_{mode}' if policy == 'pi05' else f'{task}/{mode}')
            group = dict(id=TARGETS.get(case, 'univtac_' + case), benchmark='univtac', case=case, policy=policy, task=task, modality=mode, baseline_train=str(train), baseline_feedback=str(folder), feedback_seeds=list(range(20001, 20031)), test_seeds=list(range(20031, 20061)), original_steps=17500 if policy == 'dp' else 4000, target=case in TARGETS)
            group['baseline'] = artifact(group, train / ('best.pt' if policy == 'pi05' else 'policy_best.ckpt'), 'baseline')
            groups.append(group)
    return groups

def artifact(group, path, identifier):
    path = Path(path)
    folder = path.parent
    policy = group['policy']
    return {'id': group['id'] + ':' + identifier, 'checkpoint': str(path), 'config': str(folder / ('config.yml' if policy == 'act' else 'config.json')), 'stats': str(folder / ('dataset_stats.pkl' if policy == 'act' else 'dataset_stats.npz' if policy == 'dp' else 'stats.json')), 'train_dir': str(folder)}

def checkpoints(group):
    folder = Path(group['baseline_train'])
    suffix = '*.pt' if group['policy'] == 'pi05' else '*.ckpt'
    return [artifact(group, p, 'baseline' if str(p) == group['baseline']['checkpoint'] else p.stem) for p in sorted(folder.glob(suffix)) if 'recovery' not in p.name]

def episodes(folder, seeds):
    folder = Path(folder)
    out = []
    for seed in seeds:
        p = folder / str(seed) / 'result.json'
        if not p.exists():
            raise ValueError(f'Missing seed result: {p}')
        d = read(p)
        if d['seed'] != seed:
            raise ValueError('Seed mismatch')
        worker = d.get('worker') or {}
        phase = d.get('phase', worker.get('phase'))
        status = d['status']
        good = status == 'success' and d.get('success') is True
        if d.get('success') is True and status != 'success':
            raise ValueError('Conflicting outcome')
        error = status not in ('success', 'failed')
        reason = worker.get('termination', d.get('termination'))
        termination = 'success' if good else 'exception' if error else {'native_early_stop': 'early_stop', 'action_limit': 'max_steps', 'max_steps': 'max_steps'}.get(reason)
        if termination is None:
            raise ValueError(f'Unknown termination: {reason}')
        native_path = p.parent / 'native/metadata.json'
        native = read(native_path).get(str(seed), {}) if native_path.exists() else {}
        out.append(dict(seed=seed, outcome='success' if good else ('init_error' if phase == 'reset' else 'other_error') if error else 'failure', termination=termination, steps=d.get('actions', worker.get('actions', 0)), elapsed_s=d.get('elapsed_seconds'), native={k: native[k] for k in ['rel_pose', 'inhand_bias'] if k in native}, refs={'result': str(p), 'native': str(native_path) if native_path.exists() else None, 'trajectory': str(p.parent / d['trajectory']) if d.get('trajectory') else None, 'videos': [str(p.parent / v) for v in d.get('videos', [])]}))
    return out

def curve(group, train_dir=None):
    folder = Path(train_dir or group['baseline_train'])
    metrics = folder / 'metrics.jsonl'
    points = []
    if metrics.exists():
        for i, line in enumerate(metrics.read_text().splitlines()):
            d = json.loads(line)
            loss = d.get('validation_loss', (d.get('validation') or {}).get('loss'))
            if loss is not None:
                points.append(dict(step=d.get('optimizer_steps', d.get('step')), loss=loss, source=str(metrics), line=i + 1))
    if points:
        best = min(points, key=lambda p: (p['loss'], p['step']))
        return dict(best_validation_step=best['step'], best=best, points=points, unavailable_reason=None)
    if group['policy'] == 'act' and (folder / 'train.log').exists():
        matches = re.findall('(?:val loss\\s+)([\\d.eE+-]+)\\s*(?:@|at)\\s*epoch\\s*(\\d+)', (folder / 'train.log').read_text(errors='replace'))
        if matches:
            import h5py
            split = read(folder / 'split.json')
            length = 0
            for ep in split['train_ids']:
                with h5py.File(SOURCE / 'data/official50' / group['task'] / f'episode_{ep}.hdf5', 'r') as f:
                    length += len(f['action'])
            loss, epoch = matches[-1]
            step = min((int(epoch) + 1) * math.ceil(length / 64), 4001)
            return dict(best_validation_step=step, best={'epoch': int(epoch), 'loss': float(loss), 'step': step, 'source': str(folder / 'train.log'), 'updates_per_epoch': math.ceil(length / 64)}, points=[], unavailable_reason=None)
    return dict(best_validation_step=None, best=None, points=[], unavailable_reason='No recoverable validation record')

def facts(group, artifact_id=None, feedback=None, seeds=None, train_dir=None):
    eps = episodes(feedback or group['baseline_feedback'], seeds or group['feedback_seeds'])
    c = curve(group, train_dir)
    return {'feedback_artifact_id': artifact_id or group['baseline']['id'], 'seeds': seeds or group['feedback_seeds'], 'feedback_successes': sum((e['outcome'] == 'success' for e in eps)), 'termination_counts': {k: sum((e['termination'] == k for e in eps)) for k in TERMINATIONS}, 'training_run': str(train_dir or group['baseline_train']), 'best_validation_step': c['best_validation_step'], 'unavailable_reason': c['unavailable_reason']}

def wilson(s, n):
    if n == 0:
        raise ValueError('Empty seed set')
    z = 1.959963984540054
    p = s / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0, mid - half), min(1, mid + half)]

def compare(a, b):
    aa = {r['seed']: r for r in a}
    bb = {r['seed']: r for r in b}
    if set(aa) != set(bb):
        raise ValueError('Paired comparison requires identical seed sets')
    flips = [{'seed': s, 'a': aa[s]['outcome'], 'b': bb[s]['outcome']} for s in sorted(aa) if aa[s]['outcome'] != bb[s]['outcome']]
    sa = sum((v['outcome'] == 'success' for v in aa.values()))
    sb = sum((v['outcome'] == 'success' for v in bb.values()))
    return dict(n=len(a), a_successes=sa, b_successes=sb, a_wilson95=wilson(sa, len(a)), b_wilson95=wilson(sb, len(b)), flips=flips)

def training_overview(group, train_dir):
    """Bounded train/validation/LR view; retain raw pointers and missingness."""
    folder=Path(train_dir);metrics=folder/'metrics.jsonl';rows=[]
    if metrics.exists():
        for n,line in enumerate(metrics.read_text().splitlines(),1):
            d=json.loads(line)
            rows.append({'step':d.get('optimizer_steps',d.get('step')),'train_loss':d.get('train_loss',d.get('loss')),
                         'validation_loss':d.get('validation_loss',(d.get('validation') or {}).get('loss')),
                         'lrs':d.get('lrs'),'grad_norm':d.get('grad_norm'),'source':str(metrics),'line':n})
    elif (folder/'train.log').exists():
        log=folder/'train.log'
        for n,line in enumerate(log.read_text(errors='replace').splitlines(),1):
            match=re.search(r'(\d+)/\d+.*epoch=(\d+), loss=([\d.eE+-]+)',line)
            if match:rows.append({'reported_progress':int(match[1]),'epoch':int(match[2]),'train_loss':float(match[3]),'source':str(log),'line':n})
    count=len(rows)
    indices=sorted({round(i*(count-1)/min(63,count-1)) for i in range(min(64,count))}) if count>1 else list(range(count))
    return {'records':count,'sampled_records':[rows[i] for i in indices],
            'notes':'ACT tqdm progress is a reported counter, not a verified optimizer step; complete validation series may not be recorded. Null means unrecorded.',
            'full_sources':[str(metrics if metrics.exists() else folder/'train.log')]}
