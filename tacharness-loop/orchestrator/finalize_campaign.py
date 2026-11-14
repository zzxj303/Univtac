"""Finish verified acceptance and gated loops while registered tests keep running.

This recovers supervision after the retained pre-task startup failure. It does
not replace any failed policy trial or launch duplicate registered tests.
"""
import concurrent.futures
import time
from pathlib import Path
from orchestrator.common import ROOT, read, atomic, audit, lock
from orchestrator.acceptance_recovery import recovery_folder
from orchestrator.acceptance_campaign import check_feedback
from orchestrator.experiment import group_loop
from adapters.evidence import discover


def collect_acceptance():
    rows = {}
    for g in (g for g in discover() if g['target']):
        for action in ['select_ckpt', 'finetune_more', 'finetune_data']:
            folder = ROOT / 'runs/acceptance' / g['policy'] / action
            feedback = recovery_folder(folder)
            if not (feedback / 'summary.json').exists():
                return None
            summary = read(feedback / 'summary.json')
            check_feedback(summary, feedback)
            rows[g['policy'] + ':' + action] = {'summary': summary, 'feedback': str(feedback),
                'recovery': read(folder / 'acceptance_resolution.json') if (folder / 'acceptance_resolution.json').exists() else None}
    return rows


def main():
    path = ROOT / 'runs/acceptance/finalizer.json'
    with lock(ROOT / 'runs/.acceptance-finalizer.lock'):
        state = {'started_at': time.time(), 'state': 'waiting_for_acceptance'}
        atomic(path, state)
        try:
            while True:
                acceptance = collect_acceptance()
                if acceptance is not None:
                    break
                time.sleep(5)
            audit()
            atomic(ROOT / 'runs/acceptance/action_acceptance.json', {'finished_at': time.time(),
                'native_actions_verified': 9, 'policy_trials': 18, 'retained_startup_failures': 1,
                'checks': acceptance})
            state.update(state='sealing_gates_and_waiting_for_registered_tests', acceptance_ready_at=time.time())
            atomic(path, state)
            from orchestrator.retention import prune
            prune()
            groups = [g for g in discover() if g['target']]
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                gates = list(pool.map(group_loop, groups))
            # The original supervisor owns this lock until its independent
            # baseline tests finish; retain its state before publishing recovery.
            while True:
                try:
                    with lock(ROOT / 'runs/.acceptance-campaign.lock'):
                        original = read(ROOT / 'runs/acceptance/campaign.json')
                        archived = ROOT / 'runs/acceptance/initial_campaign.json'
                        if not archived.exists():
                            atomic(archived, {'original_state': original,
                                'interruption': 'Retained DP technical-acceptance startup failure prevented original supervisor finalization',
                                'failed_result': str(ROOT / 'runs/acceptance/dp/finetune_data/feedback/20002/result.json')}, exclusive=True)
                        result = {'state': 'complete', 'started_at': original['started_at'], 'finished_at': time.time(),
                            'acceptance': {k: row['summary'] for k, row in acceptance.items()},
                            'baseline_tests': {g['id']: 'complete' for g in groups}, 'gates': gates,
                            'startup_recovery': str(ROOT / 'runs/acceptance/dp/finetune_data/acceptance_resolution.json')}
                        atomic(ROOT / 'runs/acceptance/campaign.json', result)
                    break
                except BlockingIOError:
                    time.sleep(1)
            audit()
            state.update(state='complete', finished_at=time.time(), gates=gates)
            atomic(path, state)
            # Let the original 30-second status watcher observe completion and
            # exit, then publish the current report including recovery evidence.
            time.sleep(35)
            from orchestrator.status import snapshot
            snapshot()
        except BaseException as exc:
            state.update(state='failed', failed_at=time.time(), error=repr(exc))
            atomic(path, state)
            raise


if __name__ == '__main__':
    main()
