"""Finish adapter acceptance, start each validated baseline test early, then seal gates."""
import concurrent.futures, json, time, os, sys
from pathlib import Path
from orchestrator.common import ROOT, read, atomic, audit, lock
from orchestrator.experiment import test_once, group_loop
from adapters.evidence import discover
from adapters.univtac import UniVTAC

def check_feedback(summary, folder):
    if not summary['complete'] or summary['counts']['other_error'] or summary['counts']['init_error']:
        raise RuntimeError(f'Acceptance has infrastructure/reset errors: {folder}; do not claim passed')
    for seed in [20001, 20002]:
        r = read(folder / str(seed) / 'result.json')
        if r['phase'] != 'policy' or r['actions'] <= 0:
            raise RuntimeError('Acceptance never reached policy control')

def main():
    with lock(ROOT / 'runs/.acceptance-campaign.lock'):
        groups = [g for g in discover() if g['target']]
        adapter = UniVTAC()
        for g in groups:
            for action in ['finetune_more', 'finetune_data']:
                p = ROOT / 'runs/acceptance' / g['policy'] / action / 'train/training_receipt.json'
                if not p.exists() or read(p)['actual_updates'] != 50:
                    raise RuntimeError(f'Training acceptance missing: {p}')
        state = {'started_at': time.time(), 'state': 'running', 'baseline_tests': {}, 'acceptance': {}}
        atomic(ROOT / 'runs/acceptance/supervisor.json', {'pid': os.getpid(), 'command': ' '.join([sys.executable, '-m', 'orchestrator.acceptance_campaign']), 'started_at': state['started_at']})
        atomic(ROOT / 'runs/acceptance/campaign.json', state)
        baseline_futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as test_pool:

            def validate_policy(g):
                folder = ROOT / 'runs/acceptance' / g['policy'] / 'select_ckpt/feedback'
                summary = adapter.evaluate(g, g['baseline'], [20001, 20002], folder)
                check_feedback(summary, folder)
                future = test_pool.submit(test_once, g, g['baseline'])
                return (g, future, summary)
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as selectors:
                for future in concurrent.futures.as_completed([selectors.submit(validate_policy, g) for g in groups]):
                    g, baseline, summary = future.result()
                    baseline_futures.append((g, baseline))
                    state['baseline_tests'][g['id']] = 'started'
                    state['acceptance'][g['policy'] + ':select_ckpt'] = summary
                    atomic(ROOT / 'runs/acceptance/campaign.json', state)

            def validate_trained(item):
                g, action = item
                folder = ROOT / 'runs/acceptance' / g['policy'] / action
                art = read(folder / 'artifact.json')
                from orchestrator.acceptance_recovery import recovery_folder
                feedback = recovery_folder(folder)
                summary = adapter.evaluate(g, art, [20001, 20002], feedback)
                check_feedback(summary, feedback)
                return (g['policy'] + ':' + action, summary)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as trainers:
                for future in concurrent.futures.as_completed([trainers.submit(validate_trained, (g, a)) for g in groups for a in ['finetune_more', 'finetune_data']]):
                    key, summary = future.result()
                    state['acceptance'][key] = summary
                    atomic(ROOT / 'runs/acceptance/campaign.json', state)
            for g, future in baseline_futures:
                future.result()
                state['baseline_tests'][g['id']] = 'complete'
                atomic(ROOT / 'runs/acceptance/campaign.json', state)
        gates = []
        for g in groups:
            gates.append(group_loop(g))
        state.update(state='complete', finished_at=time.time(), gates=gates)
        atomic(ROOT / 'runs/acceptance/campaign.json', state)
        audit()
if __name__ == '__main__':
    main()
