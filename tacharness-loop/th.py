import argparse, json
from pathlib import Path
from orchestrator.common import ROOT, atomic, audit, read, experiment_runs
from adapters.evidence import discover, facts, episodes, checkpoints
from orchestrator.context import build, save

def initialize(benchmark="univtac"):
    from orchestrator.common import ensure_runtime
    ensure_runtime()
    from adapters.registry import get_adapter
    adapter = get_adapter(benchmark)
    groups = adapter.discover()
    atomic(experiment_runs() / 'configs/groups.json', groups)
    rows = []
    for g in groups:
        f = adapter.facts(g, g['baseline']['id'], g['baseline_feedback'], g['feedback_seeds'], g['baseline_train'])
        summary = adapter.baseline_summary(g)
        if f['feedback_successes'] != summary['counts']['success']:
            raise ValueError('Baseline count mismatch')
        rows.append([g['id'], f['feedback_successes'], len(g['feedback_seeds'])])
        folder = experiment_runs() / g['id']
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / 'baseline.json').exists():
            atomic(folder / 'baseline.json', g['baseline'], exclusive=True)
        rounds = [dict(round=0, artifact=g['baseline'], feedback=g['baseline_feedback'], successes=f['feedback_successes'], decision='baseline')]
        if not (folder / 'rounds.json').exists():
            atomic(folder / 'rounds.json', rounds, exclusive=True)
        first = folder / 'round-1/input'
        if not first.exists():
            save(first, build(g, rounds, 1))
    (experiment_runs() / 'groups.csv').write_text('group,successes,seeds\n' + ''.join((','.join(map(str, r)) + '\n' for r in rows)))
    result = dict(groups=len(rows), episodes=sum(r[2] for r in rows), successes=sum((r[1] for r in rows)))
    atomic(experiment_runs() / 'baseline_audit.json', result)
    print(json.dumps(result))
    audit()

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    init = sub.add_parser('init')
    init.add_argument('--benchmark', default='univtac')
    sub.add_parser('status')
    sub.add_parser('audit')
    q = sub.add_parser('diagnose')
    q.add_argument('group')
    q.add_argument('--condition', choices=['raw', 'harness'], default='harness')
    q.add_argument('--label', default='attempt-1')
    tool = sub.add_parser('tool')
    tool.add_argument('name', choices=['episodes', 'curve', 'compare', 'keyframes', 'history'])
    tool.add_argument('--group', required=True)
    tool.add_argument('--round', type=int, default=1)
    tool.add_argument('--args', default='{}')
    a = p.parse_args()
    if a.cmd == 'init':
        initialize(a.benchmark)
    elif a.cmd == 'status':
        from orchestrator.status import snapshot
        print(json.dumps(snapshot(), ensure_ascii=False))
    elif a.cmd == 'audit':
        print(json.dumps(audit()))
    elif a.cmd == 'tool':
        from harness.mcp_server import Server
        server = Server(experiment_runs() / a.group / f'round-{a.round}/input/context.json', 'harness', ROOT / 'runs/cli-tools.jsonl')
        print(json.dumps(server.call('th.' + a.name, json.loads(a.args)), ensure_ascii=False))
    elif a.cmd == 'diagnose':
        from orchestrator.codex import call
        folder = experiment_runs() / a.group / 'round-1'
        r = call(folder / 'input/context.json', folder / a.condition / a.label, a.condition)
        print(json.dumps(r, ensure_ascii=False))
if __name__ == '__main__':
    main()
