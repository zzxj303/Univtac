"""Operator-only recovery of a pre-task startup failure in technical acceptance.

Never used by the formal benchmark loop. Preserves the failed attempt and reuses
all completed policy trials instead of rerunning them.
"""
import time
from pathlib import Path
from orchestrator.common import ROOT, read, atomic, audit
from adapters.evidence import discover
from adapters.univtac import UniVTAC


def recovery_folder(folder):
    resolution = folder / 'acceptance_resolution.json'
    if not resolution.exists():
        return folder / 'feedback'
    r = read(resolution)
    if r['status'] != 'passed_after_startup_recovery':
        raise RuntimeError('Unresolved acceptance startup failure')
    if r['artifact_id'] != read(folder / 'artifact.json')['id']:
        raise RuntimeError('Recovery artifact identity changed')
    failed = read(r['original_failed_result'])
    if failed['phase'] != 'startup' or failed['actions'] != 0 or failed.get('trajectory') or failed.get('videos'):
        raise RuntimeError('Recovery must never replace a policy or task trial')
    return Path(r['feedback'])


def recover_dp_data():
    g = next(g for g in discover() if g['id'] == 'univtac_hole_dp_rgbt')
    folder = ROOT / 'runs/acceptance/dp/finetune_data'
    failed_path = folder / 'feedback/20002/result.json'
    failed = read(failed_path)
    if failed['phase'] != 'startup' or failed['actions'] != 0 or failed.get('trajectory') or failed.get('videos'):
        raise RuntimeError('Not a pre-task startup failure; recovery refused')
    if failed['status'] not in ['error', 'timeout', 'interrupted']:
        raise RuntimeError('Do not repeat completed trials')
    original_first = folder / 'feedback/20001'
    first = read(original_first / 'result.json')
    if first['phase'] != 'policy' or first['actions'] <= 0:
        raise RuntimeError('Expected already completed first policy trial')
    artifact = read(folder / 'artifact.json')
    destination = folder / 'feedback-startup-recovery'
    plan_path = folder / 'startup_recovery_plan.json'
    if not plan_path.exists():
        atomic(plan_path, {'created_at': time.time(), 'scope': 'technical acceptance only; excluded from formal benchmark results',
            'original_failed_result': str(failed_path), 'artifact_id': artifact['id'],
            'reused_policy_seeds': [20001], 'new_validation_seed': 20002, 'output': str(destination)}, exclusive=True)
    probe = ROOT / 'runs/acceptance/isaac-startup-probe'
    while not (probe / 'probe_receipt.json').exists():
        time.sleep(5)
    if read(probe / 'probe_receipt.json')['exit_code'] != 0 or read(probe / 'worker_result.json')['status'] != 'startup_checked':
        raise RuntimeError('Independent startup probe did not pass; do not launch a policy trial')
    destination.mkdir(exist_ok=True)
    if not (destination / '20001').exists():
        (destination / '20001').symlink_to(original_first, target_is_directory=True)
    summary = UniVTAC().evaluate(g, artifact, [20001, 20002], destination)
    from orchestrator.acceptance_campaign import check_feedback
    check_feedback(summary, destination)
    resolution = {'status': 'passed_after_startup_recovery', 'finished_at': time.time(),
        'scope': 'technical acceptance only', 'artifact_id': artifact['id'],
        'original_failed_result': str(failed_path), 'feedback': str(destination),
        'original_startup_failures': 1, 'policy_trials': 2, 'reused_policy_seeds': [20001], 'summary': summary}
    atomic(folder / 'acceptance_resolution.json', resolution, exclusive=True)
    audit()
    return resolution


if __name__ == '__main__':
    recover_dp_data()
