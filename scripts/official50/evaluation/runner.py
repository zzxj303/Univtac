"""One attempt per seed, isolated processes, durable results and fixed denominators."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from common import CASES, HERE, ROOT, SEEDS, STATUSES, atomic_json, binding, summarize


def process_identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def terminate_group(pid):
    # Workers are session leaders; also terminate any simulator/helper descendants.
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def finish(folder, seed, status, error=None, elapsed=0, **extra):
    progress = folder / 'progress.json'
    details = json.loads(progress.read_text()) if progress.exists() else {}
    row = {'seed': seed, 'status': status, 'success': status == 'success',
           'elapsed_seconds': elapsed, 'actions': details.get('actions', 0),
           'phase': details.get('phase', 'startup'), 'error': error,
           'videos': sorted(str(p.relative_to(folder)) for p in folder.rglob('*.mp4')),
           'trajectory': 'trajectory.jsonl' if (folder/'trajectory.jsonl').exists() else None,
           **extra}
    atomic_json(folder / 'result.json', row)
    return row


def run_seed(folder, spec, seed, command, env, timeout=1800):
    folder = Path(folder)
    if (folder / 'result.json').exists():
        return json.loads((folder / 'result.json').read_text())
    if folder.exists():
        # Never reuse a claimed seed, even if interrupted before child launch.
        started_path = folder / 'started.json'
        started = json.loads(started_path.read_text()) if started_path.exists() else {}
        pid = started.get('pid')
        if pid and started.get('process_identity') and process_identity(pid) == started['process_identity']:
            terminate_group(pid)
        return finish(folder, seed, 'interrupted', 'Previous attempt has no final result; not retried',
                      elapsed=max(0, time.time()-started.get('started', time.time())))
    folder.mkdir(parents=True, exist_ok=False)
    started = {'seed': seed, 'started': time.time()}
    atomic_json(folder / 'started.json', started)
    atomic_json(folder / 'request.json', {'spec': spec, 'seed': seed, 'folder': str(folder.resolve())})
    process = None
    try:
        with (folder / 'worker.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL)
            started.update(pid=process.pid, process_identity=process_identity(process.pid))
            atomic_json(folder / 'started.json', started)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                terminate_group(process.pid)
                process.wait()
                return finish(folder, seed, 'timeout', f'Wall-clock limit {timeout}s exceeded',
                              time.time()-started['started'])
        result_path = folder / 'worker_result.json'
        if code != 0 or not result_path.exists():
            return finish(folder, seed, 'error', f'Worker exit {code}; inspect worker.log',
                          time.time()-started['started'], returncode=code)
        result = json.loads(result_path.read_text())
        if result.get('seed') != seed or result.get('status') not in STATUSES:
            raise ValueError('Worker returned an invalid result')
        return finish(folder, seed, result['status'], result.get('error'),
                      time.time()-started['started'], worker=result)
    except BaseException as exc:
        if process is not None:
            terminate_group(process.pid)
            process.wait()
        interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
        result = finish(folder, seed, 'interrupted' if interrupted else 'error', repr(exc),
                        time.time()-started['started'])
        if interrupted:
            raise
        return result
    finally:
        if process is not None:
            # A normally exited worker may have left helper processes in its group.
            terminate_group(process.pid)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--case', required=True, choices=CASES)
    parser.add_argument('--gpu', default='0', help='Physical GPU identifier exposed as cuda:0')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/official50/evaluation')
    args = parser.parse_args()
    spec = binding(args.case)
    group = args.output_root.resolve() / args.case
    group.mkdir(parents=True, exist_ok=True)
    with (group / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This evaluation group is already running; no seeds launched.')
        manifest = group / 'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != spec:
                raise SystemExit('Weights/config/source changed: refusing to mix results or repeat seeds.')
        else:
            atomic_json(manifest, spec)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=args.gpu, OMNI_KIT_ACCEPT_EULA='YES', TORCH_HOME=str(ROOT/'.cache/torch'),
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                   PYTHONUNBUFFERED='1')
        atomic_json(group/'runtime.json', {'gpu': args.gpu, 'pid': os.getpid(), 'time': time.time()})
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        try:
            for seed in SEEDS:
                folder = group / str(seed)
                command = [sys.executable, '-u', str(HERE/'worker.py'),
                           '--request', str(folder/'request.json'), '--headless', '--device', 'cuda:0']
                result = run_seed(folder, spec, seed, command, env, spec['timeout_seconds'])
                print(args.case, seed, result['status'], flush=True)
                atomic_json(group/'summary.json', summarize(group, args.case))
                if result['status'] == 'error' and result.get('phase') in ('startup', 'load_policy', 'create_environment'):
                    raise SystemExit('Infrastructure error: group stopped; attempted seed retained, no automatic retry.')
        finally:
            atomic_json(group/'summary.json', summarize(group, args.case))


if __name__ == '__main__':
    main()
