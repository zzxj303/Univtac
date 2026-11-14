from __future__ import annotations
import contextlib, ctypes, errno, fcntl, json, os, shutil, signal, subprocess, time, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('TH_UNIVTAC_ROOT', '/root/autodl-tmp/gujie/UniVTAC'))
VENDOR = ROOT / 'vendor/UniVTAC'
PYTHON = Path(os.environ.get('TH_PYTHON', str(SOURCE / '.venv/bin/python')))
PI_PYTHON = Path(os.environ.get('TH_PI_PYTHON', str(SOURCE / '.venv-pi05/bin/python')))
PROTECTED = [Path(p) for p in os.environ.get('TH_PROTECTED_ROOTS', os.pathsep.join(map(str, [SOURCE, Path('/root/autodl-tmp/gujie/lab')]))).split(os.pathsep) if p]

def read(p):
    return json.loads(Path(p).read_text())

def writable(p):
    p = Path(p).absolute()
    if not p.resolve().is_relative_to(ROOT):
        raise PermissionError(f'Output outside loop: {p}')
    return p

def atomic(p, data, exclusive=False):
    p = writable(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and p.exists():
        raise FileExistsError(p)
    tmp = p.with_name(p.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('x') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    if exclusive:
        os.link(tmp, p)
        tmp.unlink()
    else:
        os.replace(tmp, p)

def append(p, row):
    p = writable(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

@contextlib.contextmanager
def lock(p):
    p = writable(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

def check_halt():
    if (ROOT / 'runs/STOP.json').exists():
        raise RuntimeError('Run halted after a protected-directory change; inspect runs/STOP.json')

def audit():
    check_halt()
    marker = ROOT / '.start_marker'
    if not marker.exists():
        raise RuntimeError('Missing original start marker')
    cmd = ['find', *[str(p) for p in PROTECTED if p.exists()], '-newer', str(marker), '-type', 'f']
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    result = {'at': time.time(), 'command': cmd, 'changed_files': r.stdout.splitlines(), 'missing_roots': [str(p) for p in PROTECTED if not p.exists()]}
    append(ROOT / 'runs/audit.jsonl', result)
    if result['changed_files']:
        atomic(ROOT / 'runs/STOP.json', result)
        raise RuntimeError('Original directory changed; STOP: ' + str(result['changed_files'][:10]))
    return result

def environment(extra=None):
    env = os.environ.copy()
    runtime = ROOT / 'runtime'
    paths = {'TMPDIR': 'tmp', 'WANDB_DIR': 'wandb', 'XDG_CACHE_HOME': 'cache', 'XDG_CONFIG_HOME': 'config', 'XDG_DATA_HOME': 'data', 'XDG_STATE_HOME': 'state', 'TORCH_HOME': 'torch', 'TORCH_EXTENSIONS_DIR': 'torch_extensions', 'HF_HOME': 'hf', 'HF_HUB_CACHE': 'hf/hub', 'TRANSFORMERS_CACHE': 'hf/transformers', 'CUDA_CACHE_PATH': 'cuda', 'MPLCONFIGDIR': 'matplotlib', 'NUMBA_CACHE_DIR': 'numba', 'OPTIX_CACHE_PATH': 'optix', 'TRITON_CACHE_DIR': 'triton', 'CODEX_HOME': 'codex', 'JUPYTER_RUNTIME_DIR': 'jupyter', 'OMNI_USER_DIR': 'omniverse'}
    for key, sub in paths.items():
        p = runtime / sub
        p.mkdir(parents=True, exist_ok=True)
        env[key] = str(p)
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', OMNI_KIT_ACCEPT_EULA='YES', JAX_PLATFORMS='cpu')
    env['PYTHONPATH'] = os.pathsep.join(map(str, [ROOT, VENDOR, ROOT / 'vendor/OpenPI-RoboLab/src', ROOT / 'vendor/OpenPI-RoboLab/packages/openpi-client/src']))
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env

def confine_writes():
    """Landlock denies regular-file writes outside loop, independently of namespaces."""
    lib = ctypes.CDLL(None, use_errno=True)

    class Rules(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]

    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]
    abi = lib.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError('Landlock write confinement unavailable')
    mask = 1 << 1 | sum((1 << i for i in range(4, 13)))
    if abi >= 2:
        mask |= 1 << 13
    if abi >= 3:
        mask |= 1 << 14
    rules = Rules(mask)
    fd = lib.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'landlock create')
    try:
        candidates = [(ROOT, mask), (Path('/dev/shm'), mask), (Path('/proc/self/task'), 1 << 1)]
        candidates += [(p, 1 << 1) for pattern in ['null', 'zero', 'random', 'urandom', 'nvidia*', 'dri/render*', 'dri/card*'] for p in Path('/dev').glob(pattern) if not p.is_dir()]
        for p, allowed in candidates:
            parent = os.open(p, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = Beneath(allowed, parent)
                if lib.syscall(445, fd, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), 'landlock add')
            finally:
                os.close(parent)
        if lib.prctl(38, 1, 0, 0, 0) < 0 or lib.syscall(446, fd, 0) < 0:
            raise OSError(ctypes.get_errno(), 'landlock restrict')
    finally:
        os.close(fd)

def terminate(proc, grace=30):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()

def run(command, folder, env=None, timeout=None):
    check_halt()
    folder = writable(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'process.log').open('ab') as log:
        started = time.time()
        p = subprocess.Popen(command, cwd=ROOT, env=env or environment(), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, preexec_fn=confine_writes)
        atomic(folder / 'process.json', dict(command=list(map(str, command)), pid=p.pid, started_at=started))
        timed_out = False
        halted = False
        try:
            while p.poll() is None:
                if (ROOT / 'runs/STOP.json').exists():
                    halted = True
                    terminate(p)
                    break
                if timeout is not None and time.time() - started >= timeout:
                    timed_out = True
                    terminate(p)
                    break
                try:
                    p.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            terminate(p, grace=5)
            raise
        code = p.returncode
        result = dict(command=list(map(str, command)), started_at=started, finished_at=time.time(), exit_code=code, timed_out=timed_out, halted=halted)
        atomic(folder / 'process_result.json', result)
    return result

def require_vendor_imports(names):
    import sys
    result = {}
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            raise RuntimeError('Required module was not imported: ' + name)
        location = Path(module.__file__).resolve()
        if not location.is_relative_to(ROOT / 'vendor'):
            raise RuntimeError('Editable install bypassed vendor: ' + name + ' ' + str(location))
        result[name] = str(location)
    return result

def ensure_runtime():
    target = ROOT / 'runtime/io_probe.so'
    if not target.exists():
        if not shutil.which('gcc'):
            raise RuntimeError('Missing existing gcc dependency; no installation attempted')
        subprocess.run(['gcc', '-shared', '-fPIC', str(ROOT / 'adapters/carb_log_redirect.c'), '-o', str(target), '-ldl'], env=environment(), check=True)

@contextlib.contextmanager
def gpu_slot(slots=2):
    """Cross-process limit on simultaneous native simulator instances."""
    handles = []
    claimed = None
    try:
        for i in range(slots):
            p = ROOT / 'runtime' / f'gpu-slot-{i}.lock'
            p.parent.mkdir(parents=True, exist_ok=True)
            handles.append(p.open('a'))
        while claimed is None:
            check_halt()
            for f in handles:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    claimed = f
                    break
                except BlockingIOError:
                    pass
            if claimed is None:
                time.sleep(0.25)
        yield
    finally:
        for f in handles:
            f.close()


def experiment_runs(root=None):
    import re
    root = Path(root) if root is not None else ROOT
    name = os.environ.get('TH_EXPERIMENT_ID')
    if name:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name): raise ValueError('Invalid experiment ID')
        return root / 'runs/experiments' / name
    return root / 'runs'

def experiment_budget(root=None):
    root=Path(root) if root is not None else ROOT
    return experiment_runs(root) / 'configs/token_budget.json' if os.environ.get('TH_EXPERIMENT_ID') else root / 'configs/token_budget.json'
