"""Paired first-diagnosis comparison; no training and no correction retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from orchestrator.common import ROOT, read, atomic, audit, lock, experiment_runs
from orchestrator.codex import call
from orchestrator.verify import verify
from adapters.registry import get_adapter


def run(workers=2, benchmark='univtac', *, collect_only=False):
    groups = get_adapter(benchmark).discover()
    with lock(experiment_runs(ROOT) / '.diagnostic-ablation.lock'):
        def first(group):
            folder = experiment_runs(ROOT) / group['id'] / 'round-1'
            context_path = folder / 'input/context.json'
            harness_path = folder / 'harness/attempt-1/receipt.json'
            if not harness_path.exists():
                raise RuntimeError('Run initial harness diagnoses/calibration first')
            harness = read(harness_path)
            if collect_only:
                raw_path = folder / 'raw/attempt-1/receipt.json'
                if not raw_path.exists():
                    raise RuntimeError(f'Missing existing receipt: {raw_path}; offline collection cannot run Codex')
                raw = read(raw_path)
            else:
                raw = call(context_path, folder / 'raw/attempt-1', 'raw', limits={'requests': 150})
            if Path(raw['prompt_file']).read_bytes() != Path(harness['prompt_file']).read_bytes():
                raise RuntimeError('Condition prompts differ; comparison is invalid')
            rows = []
            for condition, receipt in [('harness', harness), ('raw', raw)]:
                checked = verify(read(receipt['requirement_file']), read(context_path)) if Path(receipt['requirement_file']).exists() else None
                rows.append({'group': group['id'], 'condition': condition,
                             'verification': checked,
                             'evidence_valid_rate_per_call': checked['evidence_valid'] / checked['evidence_total'] if checked and checked.get('evidence_total') else 0.0,
                             **{k: receipt.get(k) for k in
                             ['termination', 'model_requests', 'input_tokens', 'cached_input_tokens', 'output_tokens', 'elapsed_s', 'actual_model', 'actual_effort']}})
            return rows
        with ThreadPoolExecutor(max_workers=workers) as pool:
            paired = list(pool.map(first, groups))
        rows = [r for pair in paired for r in pair]
        summary = {}
        for condition in ['raw', 'harness']:
            selected = [r for r in rows if r['condition'] == condition]
            checks = [r['verification'] or {} for r in selected]
            valid = sum(v.get('evidence_valid', 0) for v in checks)
            total = sum(v.get('evidence_total', 0) for v in checks)
            summary[condition] = {'calls': len(selected), 'evidence_valid': valid, 'evidence_total': total,
                'evidence_valid_rate': valid / total if total else None,
                'mean_per_call_evidence_valid_rate': sum(r['evidence_valid_rate_per_call'] for r in selected) / len(selected),
                'fact_errors': sum(sum(k in {'feedback_successes', 'termination_counts', 'best_validation_step'} for k in v.get('fact_errors', [])) for v in checks),
                'all_fact_field_errors': sum(len(v.get('fact_errors', [])) for v in checks),
                'fact_comparable_calls': sum(bool(v.get('schema_valid')) for v in checks),
                'schema_invalid_or_missing_calls': sum(not v.get('schema_valid', False) for v in checks),
                'valid_requirements': sum(bool(v.get('passed')) for v in checks),
                'terminated_calls': sum(bool(r['termination']) for r in selected),
                **{k: sum(r[k] or 0 for r in selected) for k in ['model_requests', 'input_tokens', 'cached_input_tokens', 'output_tokens', 'elapsed_s']}}
        result = {'benchmark': benchmark, 'same_prompt_verified': True, 'corrections_included': False,
                  'fact_metric_fields': ['feedback_successes', 'termination_counts', 'best_validation_step'], 'summary': summary, 'rows': rows}
        atomic(experiment_runs(ROOT) / 'diagnostic-ablation.json', result)
        audit()
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--benchmark', default='univtac')
    p.add_argument('--collect-only', action='store_true', help='Read existing receipts only; never invoke Codex')
    a = p.parse_args()
    run(a.workers, a.benchmark, collect_only=a.collect_only)
