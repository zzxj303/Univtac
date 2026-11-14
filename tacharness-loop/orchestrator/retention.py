"""Prune only redundant intermediate acceptance checkpoints after native validation."""
import argparse
import time
from pathlib import Path
from orchestrator.common import ROOT, read, atomic, writable, audit


def plan():
    candidates = []
    for policy in ['act', 'dp', 'pi05']:
        for action in ['finetune_more', 'finetune_data']:
            folder = ROOT / 'runs/acceptance' / policy / action
            from orchestrator.acceptance_recovery import recovery_folder
            summary = recovery_folder(folder) / 'summary.json'
            if not summary.exists():
                raise RuntimeError('All six trained checkpoint feedback checks must finish before pruning')
            feedback = read(summary)
            receipt = read(folder / 'train/training_receipt.json')
            if not feedback['complete'] or feedback['counts']['other_error'] or feedback['counts']['init_error'] or not receipt['strict_reload']:
                raise RuntimeError('Acceptance did not pass; keep all checkpoints for inspection')
            final = Path(read(folder / 'artifact.json')['checkpoint']).resolve()
            for checkpoint in receipt['checkpoints']:
                path = writable(checkpoint['checkpoint'])
                if path.resolve() == final:
                    continue
                if not path.resolve().is_relative_to((folder / 'train').resolve()):
                    raise RuntimeError('Unexpected checkpoint path')
                if path.exists():
                    candidates.append({'path': str(path), 'bytes': path.stat().st_size, 'step': checkpoint['step'], 'policy': policy, 'action': action})
    return candidates


def prune():
    candidates = plan()
    journal = ROOT / 'runs/acceptance/retention.json'
    if not candidates:
        return read(journal) if journal.exists() else {'pruned': []}
    report = {'started_at': time.time(), 'reason': 'Intermediate save_at_steps checkpoints verified; final checkpoints retained', 'candidates': candidates, 'pruned': []}
    atomic(journal, report)
    for row in candidates:
        writable(row['path']).unlink()
        report['pruned'].append(row)
        atomic(journal, report)
    report['finished_at'] = time.time()
    atomic(journal, report)
    audit()
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    import json
    print(json.dumps(prune() if a.execute else plan(), indent=2))
