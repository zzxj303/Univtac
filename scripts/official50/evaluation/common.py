"""Fixed evaluation identity and auditable, simulation-free preflight."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASKS = ('insert_HDMI', 'insert_hole', 'pull_out_key')
SEEDS = list(range(20001, 20031))
CASES = {f'{policy}_{task}_{mode}': (policy, task, mode)
         for task in TASKS for policy in ('act', 'dp') for mode in ('rgb', 'rgb_tactile')}
STATUSES = ('success', 'failed', 'error', 'timeout', 'interrupted')


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit_seeds(task):
    meta = json.loads((ROOT / 'data/isaac51' / task / 'metadata.json').read_text())
    split = json.loads((ROOT / 'data/official50' / task / 'split.json').read_text())
    ids = split['train_ids'] + split['validation_ids']
    assert len(ids) == len(set(ids)) == 50 and set(ids) == set(range(50))
    excluded = sorted({int(meta[str(i)][field]) for i in ids for field in ('seed', 'source_seed')})
    assert len(SEEDS) == len(set(SEEDS)) == 30 and not set(SEEDS).intersection(excluded)
    return {'episodes': ids, 'excluded_seed_and_source_seed': excluded,
            'evaluation_seeds': SEEDS, 'overlap': []}


def binding(case):
    policy, task, mode = CASES[case]
    folder = ROOT / 'outputs/official50' / policy / task / mode
    files = {'checkpoint': folder / 'policy_best.ckpt',
             'stats': folder / ('dataset_stats.pkl' if policy == 'act' else 'dataset_stats.npz'),
             'training_config': folder / ('config.yml' if policy == 'act' else 'config.json')}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source_files = list(HERE.glob('*.py')) + [ROOT / 'task_config/clean51.yml',
        ROOT / 'envs/_base_task.py', ROOT / 'envs' / f'{task}.py',
        ROOT / 'envs/utils/env_parser.py', ROOT / 'policy/ACT/act_policy.py',
        ROOT / 'policy/ViTAL/diffusion/network.py']
    return {'case': case, 'policy': policy, 'task': task, 'modality': mode,
            'seeds': SEEDS, 'seed_audit': audit_seeds(task),
            'paths': {k: str(v) for k, v in files.items()},
            'sha256': {k: sha256(v) for k, v in files.items()},
            'task_config': 'clean51', 'expert_check': False,
            'step_limit': 600 if task == 'insert_HDMI' else 300,
            'timeout_seconds': 1800, 'dp_denoising_steps': 10,
            'dp_pred_horizon': 20, 'dp_execute_actions': 1,
            'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
            'git_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()}


def summarize(folder, case):
    rows = []
    for seed in SEEDS:
        path = Path(folder) / str(seed) / 'result.json'
        if path.exists():
            row = json.loads(path.read_text())
            if row['seed'] != seed or row['status'] not in STATUSES:
                raise ValueError(f'Invalid result: {path}')
            rows.append(row)
    counts = {status: sum(r['status'] == status for r in rows) for status in STATUSES}
    return {'case': case, 'planned': 30, 'recorded': len(rows), 'pending': 30-len(rows),
            'complete': len(rows) == 30, 'counts': counts,
            'exceptions': sum(counts[k] for k in ('error', 'timeout', 'interrupted')),
            'success_rate_fixed_denominator': counts['success']/30,
            'success_rate_is_final': len(rows) == 30, 'episodes': rows}
