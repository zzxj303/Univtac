"""Versioned diagnostic-only development runs. Never dispatches training/evaluation."""
import concurrent.futures
import json
from orchestrator.common import ROOT, read, atomic, audit, lock
from orchestrator.context import save
DEFAULT_VERSION = 'repair-v2'
from orchestrator.codex import call

GROUPS = ('univtac_hole_dp_rgbt', 'univtac_hole_pi05_rgb')

def run(group):
    folder = ROOT / 'runs/prompt-revisions' / DEFAULT_VERSION / group
    folder.mkdir(parents=True, exist_ok=True)
    with lock(folder / '.preview.lock'):
        context_path = folder / 'input/context.json'
        if not context_path.exists():
            context = read(ROOT / 'runs' / group / 'round-1/input/context.json')
            context['prompt_version'] = DEFAULT_VERSION
            save(folder / 'input', context)
        receipt = call(context_path, folder / 'harness/attempt-1', reason='prompt_development',
                       limits={'requests': 150, 'tokens': 1200000})
        audit()
        requirement_path = folder / 'harness/attempt-1/requirement.json'
        requirement = read(requirement_path) if requirement_path.exists() else None
        return {'group': group, 'receipt': str(folder / 'harness/attempt-1/receipt.json'),
                'decision': requirement['decision'] if requirement else None,
                'termination': receipt['termination'], 'verification': receipt['verification'],
                'tool_counts': receipt['tool_counts'], 'total_tokens': receipt['total_tokens']}

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, GROUPS))
    report = {'prompt_version': DEFAULT_VERSION, 'scope': 'diagnostic development only',
              'actions_executed': False, 'formal_calibration': False,
              'engineering_limits_per_call': {'requests': 150, 'tokens': 1200000},
              'results': results}
    atomic(ROOT / 'runs/prompt-revisions' / DEFAULT_VERSION / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
