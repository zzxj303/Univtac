"""Deterministic orchestration; all model output is verified before any action."""
from __future__ import annotations
import argparse, concurrent.futures, json, math, time, uuid
from pathlib import Path
from orchestrator.common import ROOT, read, atomic, lock, audit, experiment_runs, experiment_budget
from orchestrator.context import build, save, prompt_version, LEGACY_VERSION, DEFAULT_VERSION
from orchestrator.codex import call
from adapters.evidence import discover, episodes
from adapters.univtac import UniVTAC
from adapters.registry import get_adapter

def receipts(group):
    return [read(p) for p in sorted((experiment_runs(ROOT) / group['id']).glob('round-*/harness/attempt-*/receipt.json'))]

def spent(group):
    rr = receipts(group)
    started = list((experiment_runs(ROOT) / group['id']).glob('round-*/harness/attempt-*/started.json'))
    incomplete = len(started) > len(rr)
    return {'calls': max(len(rr), len(started)), 'requests': sum((r['model_requests'] or 0 for r in rr)), 'tokens': sum((r['total_tokens'] or 0 for r in rr)), 'complete': not incomplete and all((r['metering_complete'] for r in rr))}

def within(group, budget):
    s = spent(group)
    return s['complete'] and s['calls'] <= 6 and (s['requests'] <= 150) and (s['tokens'] <= budget)

def calibrate(workers=2, benchmark="univtac"):
    path = experiment_budget(ROOT)
    if path.exists():
        return read(path)
    groups = get_adapter(benchmark).discover()

    def first(g):
        folder = experiment_runs(ROOT) / g['id'] / 'round-1'
        return call(folder / 'input/context.json', folder / 'harness/attempt-1', 'harness', limits={'requests': 150})
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(first, groups))
    if any((not r['metering_complete'] for r in results)):
        raise RuntimeError('Calibration incomplete; inspect receipts, do not impute missing usage')
    versions = {r.get('prompt_version', LEGACY_VERSION) for r in results}
    if len(versions) != 1:
        raise RuntimeError('Cannot calibrate mixed prompt versions')
    values = sorted((r['total_tokens'] for r in results))
    x = 0.95 * (len(values) - 1)
    lo = math.floor(x)
    hi = math.ceil(x)
    p95 = values[lo] + (values[hi] - values[lo]) * (x - lo)
    result = {'version': 1, 'prompt_version': next(iter(versions)), 'frozen_at': time.time(), 'samples': [{k: r[k] for k in ['group', 'total_tokens', 'input_tokens', 'cached_input_tokens', 'output_tokens', 'requirement_file']} for r in results], 'n': len(results), 'quantile_method': 'linear', 'p95_single_call': p95, 'tokens_per_group': math.ceil(3 * p95 * 1.5), 'cli_calls_per_group': 6, 'model_requests_per_group': 150}
    if len(results) != 18:
        raise RuntimeError('Expected 18 initial diagnosis samples')
    atomic(path, result, exclusive=True)
    audit()
    return result

def diagnose(g, k, rounds, training_used, budget):
    folder = experiment_runs(ROOT) / g['id'] / f'round-{k}'
    input_dir = folder / 'input'
    if not input_dir.exists():
        first_input = experiment_runs(ROOT) / g['id'] / 'round-1/input/context.json'
        version = prompt_version(read(first_input)) if first_input.exists() else DEFAULT_VERSION
        save(input_dir, build(g, rounds, k, training_used, prompt_version=version))
    previous = None
    for attempt in (1, 2):
        target = folder / 'harness' / f'attempt-{attempt}'
        receipt_path = target / 'receipt.json'
        if receipt_path.exists():
            r = read(receipt_path)
            if not r['termination'] and Path(r['requirement_file']).exists():
                from orchestrator.verify import verify
                r['verification'] = verify(read(r['requirement_file']), read(input_dir / 'context.json'))
                atomic(target / 'verification-current.json', r['verification'])
        else:
            s = spent(g)
            if not s['complete'] or s['calls'] >= 6 or s['requests'] >= 150 or (s['tokens'] >= budget):
                return (None, 'budget_exhausted')
            correction = None
            reason = 'initial'
            if previous:
                if previous['termination'] in ('isolation_violation', 'metering_incomplete', 'model_mismatch'):
                    return (None, previous['termination'])
                if previous['termination']:
                    reason = 'retry_error'
                else:
                    reason = 'correction'
                    correction = {'requirement': read(previous['requirement_file']), 'rejection': previous['verification']}
            r = call(input_dir / 'context.json', target, 'harness', reason, correction, {'requests': 150 - s['requests'], 'tokens': budget - s['tokens']})
        if r['termination'] in ('isolation_violation', 'metering_incomplete', 'model_mismatch'):
            return (None, r['termination'])
        if not within(g, budget):
            return (None, 'budget_exceeded')
        if not r['termination'] and r['verification'] and r['verification']['passed']:
            requirement = read(r['requirement_file'])
            if requirement.get('repair_intent'):
                if attempt != 1: return (None, 'repair_requires_second_cli_slot')
                s = spent(g)
                second = folder / 'harness/attempt-2'
                if not (second / 'receipt.json').exists() and (s['calls'] >= 6 or s['requests'] >= 150 or s['tokens'] >= budget): return (None, 'budget_exhausted')
                from orchestrator.repair_stage import execute
                req, error = execute(g, k, input_dir, requirement, second, {'requests':150-s['requests'],'tokens':budget-s['tokens']})
                if not within(g,budget): return (None,'budget_exceeded')
                return req,error
            return (requirement, None)
        previous = r
    return (None, 'diagnosis_failed')

def test_once(group, art, gate=None):
    """Artifact-ID registry; a baseline alias never launches another test."""
    root = ROOT / 'runs/private_test' / group['id']
    root.mkdir(parents=True, exist_ok=True)
    registry = root / 'registry.json'
    with lock(root / '.registry.lock'):
        entries = read(registry) if registry.exists() else []
        identity = {'artifact_id': art['id'], 'stats': art['stats'], 'config': art['config'], 'seeds': group['test_seeds'], 'protocol': get_adapter(group).protocol}
        match = next((e for e in entries if e['identity'] == identity), None)
        if match is None:
            if gate is None and art['id'] != group['baseline']['id']:
                raise ValueError('Only baseline test may precede gate')
            match = {'identity': identity, 'run_id': str(uuid.uuid4()), 'directory': str(root / ('baseline' if art['id'] == group['baseline']['id'] else str(uuid.uuid4()))), 'registered_at': time.time(), 'gate_id': gate['gate_id'] if gate else None, 'baseline_early': gate is None}
            entries.append(match)
            atomic(registry, entries)
    out = Path(match['directory'])
    if (out / 'summary.json').exists():
        return {'reused': True, 'registry': match, 'result': read(out / 'summary.json')}
    while True:
        try:
            with lock(root / (match['run_id'] + '.lock')):
                if (out / 'summary.json').exists():
                    return {'reused': True, 'registry': match, 'result': read(out / 'summary.json')}
                out.mkdir(parents=True, exist_ok=True)
                if not (out / 'test_started.json').exists():
                    now = time.time()
                    if gate and (not match['baseline_early']) and (now <= gate['sealed_at']):
                        raise RuntimeError('Test must follow gate sealing')
                    atomic(out / 'test_started.json', {'started_at': now, 'gate_id': match['gate_id'], 'baseline_early': match['baseline_early']}, exclusive=True)
                result = get_adapter(group).evaluate(group, art, group['test_seeds'], out)
                return {'reused': False, 'registry': match, 'result': result}
        except BlockingIOError:
            time.sleep(1)

def baseline_tests(workers=2, benchmark="univtac"):
    groups = [g for g in get_adapter(benchmark).discover() if g['target']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda g: test_once(g, g['baseline']), groups))

def group_loop(g):
    calibration = read(experiment_budget(ROOT))
    budget = calibration['tokens_per_group']
    root = experiment_runs(ROOT) / g['id']
    adapter = get_adapter(g)
    with lock(root / '.loop.lock'):
        if (root / 'gate.json').exists():
            gate = read(root / 'gate.json')
        else:
            first_input = root / 'round-1/input/context.json'
            if first_input.exists() and prompt_version(read(first_input)) != calibration.get('prompt_version', LEGACY_VERSION):
                raise RuntimeError('Prompt version differs from frozen calibration; calibrate the new experiment separately')
            rounds = read(root / 'rounds.json')
            training_used = sum((r.get('training_updates', 0) for r in rounds))
            stop_reason = 'round_limit'
            for k in range(len(rounds), 4):
                committed = root / f'round-{k}/round.json'
                if committed.exists():
                    row = read(committed)
                    if row['round'] != k:
                        raise RuntimeError('Round commit/index disagreement')
                    rounds.append(row)
                    training_used += row.get('training_updates', 0)
                    atomic(root / 'rounds.json', rounds)
                    audit()
                    continue
                req, error = diagnose(g, k, rounds, training_used, budget)
                if error:
                    stop_reason = error
                    break
                if req['decision'] == 'stop':
                    stop_reason = 'agent_stop'
                    stop_path = root / f'round-{k}/stop.json'
                    if stop_path.exists():
                        if read(stop_path) != req:
                            raise RuntimeError('Committed stop requirement changed')
                    else:
                        atomic(stop_path, req, exclusive=True)
                    audit()
                    break
                context = read(root / f'round-{k}/input/context.json')
                artifacts = {key: a for a in context['artifacts'] for key in (a['id'], a['checkpoint'])}
                folder = root / f'round-{k}'
                if req['decision'] == 'select_ckpt':
                    candidates = [artifacts[i] for i in req['parameters']['checkpoint_ids']]
                    updates = 0
                else:
                    updates = req['parameters']['steps']
                    new = adapter.train(g, artifacts[req['init_from']], req['parameters'], folder / 'training')
                    candidates = [new]
                    training_used += updates
                evaluations = []
                for index, a in enumerate(candidates):
                    previous = next((old for old in rounds if old['artifact']['id'] == a['id']), None)
                    if previous:
                        evaluations.append({'artifact': a, 'feedback': previous['feedback'], 'successes': previous['successes'], 'reused_feedback': True})
                    else:
                        destination = folder / 'feedback' / f'candidate-{index + 1}'
                        summary = adapter.evaluate(g, a, g['feedback_seeds'], destination)
                        evaluations.append({'artifact': a, 'feedback': str(destination), 'successes': summary['successes'], 'reused_feedback': False})
                selected = max(evaluations, key=lambda e: e['successes'])
                row = {'round': k, **selected, 'decision': req['decision'], 'training_updates': updates, 'eligible': within(g, budget), 'completed_at': time.time(), 'candidates': evaluations, 'requirement':req}
                if (folder / 'repair-input/context.json').exists():
                    row['repair_session'] = read(folder / 'repair-input/context.json')['repair_session']
                atomic(folder / 'round.json', row, exclusive=True)
                rounds.append(row)
                atomic(root / 'rounds.json', rounds)
                audit()
            eligible = [r for r in rounds if r.get('eligible', True)]
            best = max(eligible, key=lambda r: (r['successes'], -r['round']))
            gate = {'gate_id': str(uuid.uuid4()), 'group': g['id'], 'selected_round': best['round'], 'artifact': best['artifact'], 'feedback_successes': best['successes'], 'stop_reason': stop_reason, 'spent': spent(g), 'sealed_at': time.time()}
            atomic(root / 'gate.json', gate, exclusive=True)
            audit()
        result = test_once(g, gate['artifact'], gate)
        atomic(root / 'test_receipt.json', result)
        return gate

def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['calibrate', 'baseline-test', 'loop', 'all'])
    p.add_argument('--group')
    p.add_argument('--benchmark', default='univtac')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    if a.command == 'calibrate':
        print(json.dumps(calibrate(a.workers, a.benchmark)))
    elif a.command == 'baseline-test':
        print(json.dumps(baseline_tests(a.workers, a.benchmark)))
    else:
        if a.command == 'all':
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                baseline = pool.submit(baseline_tests, a.workers, a.benchmark)
                calibrate(a.workers, a.benchmark)
                baseline.result()
        groups = [g for g in get_adapter(a.benchmark).discover() if g['target'] and (not a.group or g['id'] == a.group)]
        for g in groups:
            print(json.dumps(group_loop(g)), flush=True)
if __name__ == '__main__':
    main()
