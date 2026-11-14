from __future__ import annotations
import json, os, shutil, time, uuid
from pathlib import Path
from orchestrator.common import ROOT, SOURCE, VENDOR, PYTHON, PI_PYTHON, read, atomic, environment, run, lock, audit, writable, gpu_slot
from adapters.evidence import discover, episodes, checkpoints, artifact, facts, curve

class UniVTAC:
    discover = staticmethod(discover)
    checkpoints = staticmethod(checkpoints)

    protocol = 'univtac-clean51-loop-v1'
    episodes = staticmethod(episodes)
    facts = staticmethod(facts)
    curve = staticmethod(curve)

    def demonstration_files(self, group):
        from adapters.datasets import source_files
        return source_files(group)

    def demonstrations(self, group):
        from adapters.datasets import catalog
        return catalog(group)

    def prepare_repairs(self, group, intents, folder, parent=None):
        from adapters.repairs import prepare
        return prepare(group, intents, folder, parent)

    def try_repair(self, session, demo_id, patch):
        from adapters.repairs import try_repair
        return try_repair(session, demo_id, patch)

    def verify_repair(self, session, demo_id, candidate):
        from adapters.repairs import verify_repair
        return verify_repair(session, demo_id, candidate)

    def accepted_repairs(self, session):
        from adapters.repairs import accepted_replacements
        return accepted_replacements(session)

    def create_dataset_version(self, group, ids, replacements=None, parent=None):
        from adapters.datasets import create_version
        return create_version(group, ids, replacements, parent)

    def baseline_summary(self, group):
        return read(Path(group['baseline_feedback']) / 'summary.json')

    def raw_files(self, group, rounds):
        current = rounds[-1]
        files = []
        records = []
        for r in rounds:
            if r.get('repair_session'):
                from adapters.repairs import repair_sources
                files.extend(repair_sources(r['repair_session']))
            records.extend([r] + r.get('candidates', []))
        folders = {Path(group['baseline_train'])} | {Path(r['artifact']['train_dir']) for r in records}
        for folder in folders:
            for p in folder.iterdir():
                if p.is_file() and p.suffix in ('.json', '.jsonl', '.log', '.yml', '.yaml', '.txt'):
                    files.append(str(p.resolve()))
        for r in records:
            folder = Path(r['feedback'])
            for name in ['summary.json', 'manifest.json']:
                if (folder / name).is_file(): files.append(str((folder / name).resolve()))
            for seed in group['feedback_seeds']:
                result = folder / str(seed) / 'result.json'
                if not result.is_file(): continue
                files.append(str(result.resolve()))
                d = read(result)
                for rel in [d.get('trajectory'), 'native/metadata.json']:
                    if not rel: continue
                    p = result.parent / rel
                    if p.is_file() and p.resolve().is_relative_to(folder.resolve()): files.append(str(p.resolve()))
        return sorted(set(files))

    def validate_training_data(self, group, ids):
        errors = []
        split = read(SOURCE / 'data/official50' / group['task'] / 'split.json')
        if not ids or len(set(ids)) != len(ids) or any(type(i) != int or not 0 <= i < 100 for i in ids) or set(ids) & set(split['validation_ids']):
            errors.append('Invalid training demonstration IDs')
            return errors
        meta = read(SOURCE / 'data/isaac51' / group['task'] / 'metadata.json')
        if not all(str(i) in meta for i in ids + split['validation_ids']):
            return ['Demonstration seed metadata missing']
        demo = {int(meta[str(i)][k]) for i in ids + split['validation_ids'] for k in ['seed', 'source_seed']}
        if demo & set(group['feedback_seeds'] + group['test_seeds']):
            errors.append('Demonstration/evaluation seed overlap')
        return errors

    def train(self, group, source, request, output, *, save_at_steps=None):
        out = writable(output)
        out.mkdir(parents=True, exist_ok=True)
        steps = request['steps']
        if save_at_steps is not None and (save_at_steps != sorted(set(save_at_steps)) or any((type(n) != int or n < 1 or n > steps for n in save_at_steps))):
            raise ValueError('Invalid save_at_steps')
        identity = dict(group=group, artifact=source, output=str(out / 'train'), steps=steps, lr_scale=request.get('lr_scale'), train_ids=request.get('train_ids'), dataset_version=request.get('dataset_version') or source.get('dataset_version'), save_at_steps=save_at_steps)
        if (out / 'request.json').exists() and read(out / 'request.json') != identity:
            raise RuntimeError('Training identity changed; cannot reuse this output directory')
        if (out / 'artifact.json').exists():
            return read(out / 'artifact.json')
        if (out / 'train/metrics.jsonl').exists() or (out / 'process/process.json').exists():
            raise RuntimeError('Interrupted training requires reconciliation; no silent restart')
        atomic(out / 'request.json', identity)
        python = PI_PYTHON if group['policy'] == 'pi05' else PYTHON
        with lock(ROOT / 'runs/gpu-training.lock'):
            result = run([str(python), '-m', 'adapters.train_worker', '--request', str(out / 'request.json')], out / 'process', timeout=12 * 3600)
        audit()
        if result['exit_code'] != 0:
            raise RuntimeError(f'Training failed; inspect {out}/process/process.log')
        receipt = read(out / 'train/training_receipt.json')
        if receipt['actual_updates'] != steps:
            raise RuntimeError('Wrong optimizer update count')
        a = artifact(group, receipt['checkpoints'][-1]['checkpoint'], str(out.relative_to(ROOT / 'runs')))
        if receipt.get('dataset_version'): a['dataset_version'] = receipt['dataset_version']
        atomic(out / 'artifact.json', a, exclusive=True)
        return a

    def evaluate(self, group, art, seeds, output):
        rules = read(ROOT / 'configs/evaluation-protocols.json')
        if group['case'] not in rules:
            raise RuntimeError('Reset retry rule not verified for this group; inspect baseline records and ask the user if unavailable')
        retries = rules[group['case']]['reset_retries']
        out = writable(output)
        out.mkdir(parents=True, exist_ok=True)
        with lock(out / '.lock'):
            if len(seeds) != len(set(seeds)):
                raise ValueError('Duplicate seeds')
            permitted = set(group['feedback_seeds']) | set(group['test_seeds'])
            if not set(seeds) <= permitted:
                raise ValueError('Unregistered evaluation seeds')
            split = read(Path(art['train_dir']) / 'split.json') if (Path(art['train_dir']) / 'split.json').exists() else read(SOURCE / 'data/official50' / group['task'] / 'split.json')
            metadata = read(SOURCE / 'data/isaac51' / group['task'] / 'metadata.json')
            demo = {int(metadata[str(i)][k]) for i in split['train_ids'] + split['validation_ids'] for k in ['seed', 'source_seed']}
            if demo & set(seeds):
                raise ValueError('Demonstration/evaluation seed overlap')
            spec = dict(case=group['case'], task=group['task'], policy=group['policy'], modality=group['modality'], seeds=seeds, paths={'checkpoint': art['checkpoint'], 'training_config': art['config'], 'stats': art['stats']}, step_limit=600 if group['task'] == 'insert_HDMI' else 300, timeout_seconds=1800, artifact_id=art['id'], dp_denoising_steps=10, dp_pred_horizon=20, dp_execute_actions=1)
            if (out / 'manifest.json').exists() and read(out / 'manifest.json') != spec:
                raise RuntimeError('Evaluation identity changed')
            if not (out / 'manifest.json').exists():
                atomic(out / 'manifest.json', spec, exclusive=True)
            if (out / 'summary.json').exists():
                return read(out / 'summary.json')
            for seed in seeds:
                if shutil.disk_usage(ROOT).free < 8 * 2 ** 30:
                    raise RuntimeError('Disk reserve below 8 GiB; no episode launched')
                result_path = out / str(seed) / 'result.json'
                if result_path.exists():
                    continue
                final = None
                for attempt in range(1, retries + 2):
                    folder = out / str(seed) / f'attempt-{attempt}'
                    if (folder / 'result.json').exists():
                        row = read(folder / 'result.json')
                    else:
                        if (folder / 'process/process.json').exists():
                            raise RuntimeError(f'Interrupted seed attempt: {folder}; cannot silently repeat')
                        folder.mkdir(parents=True, exist_ok=True)
                        atomic(folder / 'request.json', {'spec': spec, 'seed': seed, 'folder': str(folder)})
                        worker = VENDOR / 'scripts' / ('pi05_official50' if group['policy'] == 'pi05' else 'official50') / 'evaluation/worker.py'
                        (folder / 'extension-logs').mkdir(exist_ok=True)
                        env = environment({'TH_OMNI_LOG_DIR': folder / 'extension-logs', 'LD_PRELOAD': ROOT / 'runtime/io_probe.so', 'TH_OUTPUT_ROOT': folder, 'TH_SPLIT': SOURCE / 'data/official50' / group['task'] / 'split.json', 'TH_STATS': art['stats'], 'TH_DATA_DIR': SOURCE / 'data/official50' / group['task']})
                        python = PI_PYTHON if group['policy'] == 'pi05' else PYTHON
                        with gpu_slot():
                            result = run([str(python), str(worker), '--request', str(folder / 'request.json'), '--headless', '--device', 'cuda:0'], folder / 'process', env, 1800)
                        worker_result = read(folder / 'worker_result.json') if (folder / 'worker_result.json').exists() else {}
                        progress = read(folder / 'progress.json') if (folder / 'progress.json').exists() else {}
                        status = worker_result.get('status', 'error')
                        if result['timed_out']:
                            status = 'timeout'
                        elif result['exit_code'] != 0:
                            status = 'error'
                        row = dict(seed=seed, status=status, success=status == 'success', phase=worker_result.get('phase', progress.get('phase', 'startup')), actions=worker_result.get('actions', progress.get('actions', 0)), elapsed_seconds=result['finished_at'] - result['started_at'], error=worker_result.get('error') or (f"exit {result['exit_code']}" if result['exit_code'] else None), worker=worker_result, videos=[str(p.relative_to(folder)) for p in folder.rglob('*.mp4')], trajectory='trajectory.jsonl' if (folder / 'trajectory.jsonl').exists() else None, attempt=attempt)
                        atomic(folder / 'result.json', row, exclusive=True)
                        audit()
                    final = row
                    if not (attempt <= retries and row['status'] in ('error', 'timeout', 'interrupted') and (row['phase'] == 'reset') and (row['actions'] == 0)):
                        break
                selected = out / str(seed) / f"attempt-{final['attempt']}"
                parent = out / str(seed)
                final = {**final, 'videos': [f"attempt-{final['attempt']}/" + v for v in final['videos']], 'trajectory': f"attempt-{final['attempt']}/trajectory.jsonl" if final['trajectory'] else None}
                if (selected / 'native').exists() and (not (parent / 'native').exists()):
                    (parent / 'native').symlink_to(selected / 'native', target_is_directory=True)
                atomic(result_path, final, exclusive=True)
                if final['status'] in ('error', 'timeout', 'interrupted') and final['phase'] not in ('reset', 'policy'):
                    raise RuntimeError(f'Infrastructure failure at {selected}; stopped without retry')
            eps = episodes(out, seeds)
            summary = {'complete': True, 'artifact_id': art['id'], 'seeds': seeds, 'successes': sum((e['outcome'] == 'success' for e in eps)), 'counts': {k: sum((e['outcome'] == k for e in eps)) for k in ['success', 'failure', 'init_error', 'other_error']}, 'finished_at': time.time()}
            atomic(out / 'summary.json', summary, exclusive=True)
            audit()
            return summary
