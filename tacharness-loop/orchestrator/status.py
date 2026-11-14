"""Read receipts into a shareable status report; never infer success from code presence."""
import argparse
import time
from pathlib import Path
from orchestrator.common import ROOT, read, atomic, audit


def snapshot():
    def optional(path, default=None):
        return read(path) if path.exists() else default
    training = []
    feedback = []
    for policy in ['act', 'dp', 'pi05']:
        for action in ['select_ckpt', 'finetune_more', 'finetune_data']:
            folder = ROOT / 'runs/acceptance' / policy / action
            rec = optional(folder / 'train/training_receipt.json')
            if rec:
                training.append({'policy': policy, 'action': action, 'receipt': str((folder / 'train/training_receipt.json').relative_to(ROOT)),
                                 **{k: rec.get(k) for k in ['actual_updates', 'strict_reload', 'parameter_sample_changed', 'seconds', 'peak_cuda_bytes']}})
            resolution = optional(folder / 'acceptance_resolution.json')
            feedback_folder = Path(resolution['feedback']) if resolution else folder / 'feedback'
            summary = optional(feedback_folder / 'summary.json')
            completed = list(feedback_folder.glob('*/result.json'))
            feedback.append({'policy': policy, 'action': action, 'complete': bool(summary and summary['complete']),
                             'completed_seeds': len(completed), 'expected_seeds': 2,
                             'summary': summary, 'startup_recovery': resolution})
    diagnostics = []
    for condition in ['harness', 'raw']:
        initial = [read(p) for p in (ROOT / 'runs').glob(f'univtac*/round-1/{condition}/attempt-1/receipt.json')]
        diagnostics.append({'condition': condition, 'first_calls': len(initial),
                            'metered_calls': sum(r['metering_complete'] for r in initial),
                            'actual_models': sorted({r['actual_model'] for r in initial if r['actual_model']}),
                            'actual_efforts': sorted({r['actual_effort'] for r in initial if r['actual_effort']})})
    tests = []
    for folder in sorted((ROOT / 'runs/private_test').glob('*/baseline')):
        results = [read(p) for p in folder.glob('*/result.json')]
        tests.append({'group': folder.parent.name, 'completed_seeds': len(results), 'expected_seeds': 30,
                      'successes_so_far': sum(r['success'] for r in results),
                      'complete': (folder / 'summary.json').exists()})
    gates = [read(p) for p in (ROOT / 'runs').glob('univtac*/gate.json')]
    logs = ROOT / 'runs/audit.jsonl'
    latest_audit = None
    if logs.exists():
        import json
        latest_audit = json.loads(logs.read_text().splitlines()[-1])
    budget = optional(ROOT / 'configs/token_budget.json', {})
    campaign = optional(ROOT / 'runs/acceptance/campaign.json', {})
    result = {'updated_at': time.time(), 'implementation': 'implemented',
              'experiment_state': campaign.get('state', 'not_started'),
              'baseline_parser': optional(ROOT / 'runs/baseline_audit.json'),
              'unit_tests': optional(ROOT / 'runs/unit-tests.json'),
              'token_budget': {k: budget.get(k) for k in ['n', 'p95_single_call', 'tokens_per_group', 'frozen_at']},
              'diagnostics': diagnostics, 'training_acceptance': training, 'feedback_acceptance': feedback,
              'private_tests': tests, 'gates': gates, 'last_readonly_audit': latest_audit,
              'diagnostic_ablation': optional(ROOT / 'runs/diagnostic-ablation.json', {}).get('summary'),
              'libero': 'Interface only; no remote server accessed', 'evolution': 'excluded', 'checksums': 'not used'}
    atomic(ROOT / 'STATUS.json', result)
    lines = ['# 验收记录', '', '此文件由 `python -m orchestrator.status` 从真实收据生成。未完成的评测不作为实验结论。', '',
             f"运行状态：`{result['experiment_state']}`。", '',
             '## 本机 LLM', '', '| 条件 | 首次调用 | 完整计量 | 实际模型 | 推理强度 |', '|---|---:|---:|---|---|']
    for row in diagnostics:
        lines.append(f"| {row['condition']} | {row['first_calls']}/18 | {row['metered_calls']} | {', '.join(row['actual_models'])} | {', '.join(row['actual_efforts'])} |")
    lines += ['', f"冻结预算：每组 {budget.get('tokens_per_group', '待定')} tokens；6 次 CLI；150 次模型请求。", '',
              '## 训练与推理验收', '', '| 策略 | 动作 | 训练更新 | 反馈回合 | 完成 |', '|---|---|---:|---:|---|']
    for row in feedback:
        rec = next((r for r in training if r['policy'] == row['policy'] and r['action'] == row['action']), {})
        lines.append(f"| {row['policy']} | {row['action']} | {rec.get('actual_updates', '—')} | {row['completed_seeds']}/2 | {row['complete']} |")
    lines += ['', '## 正式 test', '', '| 目标组 | 已完成 | 完成整组 |', '|---|---:|---|']
    for row in tests:
        lines.append(f"| {row['group']} | {row['completed_seeds']}/30 | {row['complete']} |")
    lines += ['', '## 门控', '']
    for gate in gates:
        lines.append(f"- {gate['group']}：第 {gate['selected_round']} 轮；停止原因 `{gate['stop_reason']}`。")
    if not gates:
        lines.append('尚未生成门控结果。')
    lines += ['', '## 边界与限制', '',
              '- 原目录按 `.start_marker` 检查；不使用 SHA-256 或其他摘要。',
              '- `/dev/shm` 仅用于用户许可的临时仿真 IPC。',
              '- LIBERO 仅提供适配器接口；进化已移除。',
              '- 事实错误或无效证据会阻止动作；模型输出不等于已通过验收。',
              '- `runs/`、`runtime/` 内含私有运行状态，不应随开源代码发布。', '']
    (ROOT / 'VALIDATION.md').write_text('\n'.join(lines))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        result = snapshot()
        if not args.watch or result['experiment_state'] == 'complete':
            break
        time.sleep(30)
