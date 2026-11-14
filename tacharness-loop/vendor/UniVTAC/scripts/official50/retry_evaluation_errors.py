"""Explicit, single replacement attempt for the frozen ACT/DP exception list."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / 'scripts/official50/evaluation'
sys.path.insert(0, str(EVAL))
from common import CASES, SEEDS, atomic_json, sha256, summarize
from runner import run_seed, finish, process_identity, terminate_group

EXCEPTIONS = {'error', 'timeout', 'interrupted'}
README = ROOT.parent / 'README-six-task-rsi.md'
BASE = ROOT / 'outputs/official50/evaluation'


def read(path):
    return json.loads(path.read_text())


def update_readme(campaign, jobs, state):
    summaries = {case: summarize(BASE / case, case) for case in CASES}
    totals = {k: sum(s['counts'][k] for s in summaries.values())
              for k in ('success', 'failed', 'error', 'timeout', 'interrupted')}
    errors = sum(totals[k] for k in EXCEPTIONS)
    done = sum(j.get('published', False) for j in jobs)
    text = README.read_text()
    # Match the destination URL so pi0.5 and historical LIBERO cells are untouched.
    for case, summary in summaries.items():
        count = summary['counts']['success']
        pattern = r'\[[^\]\n]+\](\(UniVTAC/outputs/official50/evaluation/' + re.escape(case) + r'/summary\.json\))'
        text, n = re.subn(pattern, lambda m: f'[{100*count/30:.1f}%（{count}/30）]' + m[1], text)
        if n != 1:
            raise RuntimeError(f'Expected exactly one README result cell for {case}, found {n}')
    header = '### official50 正式评测结果'
    start = text.index(header)
    end = text.index('\n## ', start)
    section = text[start:end]
    section = section.replace('此次串行补跑保持', '此次补跑（最多两回合并行）保持')
    section = re.sub(r'^### official50 正式评测结果[^\n]*',
                     '### official50 正式评测结果（含用户指定的异常回合补跑）', section)
    section = section.replace('每组 seed `20001–20030` 各一次', '每组固定 seed `20001–20030`')
    section = section.replace('不进行专家筛选、不重试或替换失败 seed；异常和中断保留并计入固定分母 30。',
        '不进行专家筛选，不补跑正常任务失败回合。本次按用户要求仅将原有异常回合以相同 seed 各补跑一次，并以新结果替换当前统计；原始记录单独备份。补跑后仍异常的回合保留，计入固定分母 30。')
    section = re.sub(r'\*\*总计：[^\n]+',
        f'**当前总计：{totals["success"]} 次成功、{totals["failed"]} 次任务失败、{errors} 次运行异常，共 360 回合。**', section)
    section = re.sub(r'\*\*当前总计：[^\n]+',
        f'**当前总计：{totals["success"]} 次成功、{totals["failed"]} 次任务失败、{errors} 次运行异常，共 360 回合。**', section)
    for task, label in [('insert_HDMI', 'HDMI 插接'), ('insert_hole', '斜孔插入'), ('pull_out_key', '旋转并拔出钥匙')]:
        counts = [summaries[f'{p}_{task}_{m}']['exceptions']
                  for p in ('act', 'dp') for m in ('rgb', 'rgb_tactile')]
        section = re.sub(r'^\| ' + label + r' \|[^\n]+',
                         '| ' + label + ' | ' + ' | '.join(map(str, counts)) + ' |', section, flags=re.M)
    section = re.sub(r'21 次异常全部发生在初始化阶段：[^\n]+',
        '原始 21 次异常均发生在初始化阶段：19 次 reset 超过 120 秒，2 次触觉 marker 稳定等待超时。此次补跑（最多两回合并行）保持原有时间限制、物理配置、权重、成功条件与动作上限不变；每个目标回合只补跑一次，不反复尝试到成功。当前结果采用补跑结果，不能表述为未经重试的单次评测。π0.5 结果未补跑。', section)
    marker_start = '<!-- ACT_DP_EXCEPTION_RERUN_START -->'
    marker_end = '<!-- ACT_DP_EXCEPTION_RERUN_END -->'
    relative = campaign.relative_to(ROOT.parent)
    note = (f'{marker_start}\n'
            f'异常补跑状态：**{state}，已替换 {done}/{len(jobs)} 回合**。'
            '原始结果为 47 成功／292 任务失败／21 异常。'
            '补跑进行中时，表格为已替换结果与尚未补跑原结果的合并统计；每组分母始终为 30。'
            f' [补跑进度与清单]({relative}/status.json)、[原始备份]({relative}/original/)。\n'
            f'{marker_end}')
    if marker_start in section:
        section = re.sub(re.escape(marker_start) + r'.*?' + re.escape(marker_end), lambda _: note, section, flags=re.S)
    else:
        section += '\n' + note + '\n'
    text = text[:start] + section + text[end:]
    temp = README.with_name(README.name + '.rerun.tmp')
    temp.write_text(text)
    os.replace(temp, README)
    atomic_json(campaign / 'aggregate.json', {'totals': totals, 'exceptions': errors,
        'denominator': 360, 'replaced': done, 'target': len(jobs), 'state': state})


def verify(spec):
    for key, digest in spec['sha256'].items():
        assert sha256(spec['paths'][key]) == digest, f'Changed artifact: {key}'
    for name, digest in spec['source_sha256'].items():
        assert sha256(ROOT / name) == digest, f'Changed evaluation source: {name}'


def publish(campaign, job):
    case, seed = job['case'], job['seed']
    source = campaign / 'attempts' / case / str(seed)
    target = BASE / case / str(seed)
    backup = campaign / 'original' / case / str(seed)
    staging = target.with_name(str(seed) + '.replacement')
    # Recovery from a crash between archiving the old directory and publishing.
    if backup.exists() and target.exists():
        assert sha256(target / 'result.json') == sha256(source / 'result.json')
    else:
        if not backup.exists():
            assert sha256(target / 'result.json') == job['original_sha256']
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(source, staging)
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            os.rename(target, backup)
        os.rename(staging, target)
    job['published'] = True
    job['result'] = read(target / 'result.json')['status']
    atomic_json(BASE / case / 'summary.json', summarize(BASE / case, case))


def execute(campaign, job, env):
    case, seed = job['case'], job['seed']
    spec = read(BASE / case / 'manifest.json')
    verify(spec)
    folder = campaign / 'attempts' / case / str(seed)
    command = [str(ROOT / '.venv/bin/python'), '-u', str(EVAL / 'worker.py'),
               '--request', str(folder / 'request.json'), '--headless', '--device', 'cuda:0']
    # Adopt an existing isolated worker during a scheduler handoff, never rerun it.
    if folder.exists() and not (folder / 'result.json').exists() and (folder / 'started.json').exists():
        started = read(folder / 'started.json')
        pid = started.get('pid')
        identity = started.get('process_identity')
        def alive():
            if not pid or not identity or process_identity(pid) != identity:
                return False
            try:
                return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
            except FileNotFoundError:
                return False
        if alive():
            argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            assert str(folder / 'request.json').encode() in argv, 'Worker identity mismatch'
        while alive():
            if time.time() - started['started'] >= spec['timeout_seconds']:
                terminate_group(pid)
                return finish(folder, seed, 'timeout', 'Wall-clock limit 1800s exceeded',
                              time.time() - started['started'])
            time.sleep(1)
        result_path = folder / 'worker_result.json'
        if result_path.exists():
            row = read(result_path)
            assert row['seed'] == seed and row['status'] in ('success', 'failed', 'error')
            return finish(folder, seed, row['status'], row.get('error'),
                          time.time() - started['started'], worker=row)
        return finish(folder, seed, 'interrupted', 'Existing attempt exited without worker result',
                      time.time() - started['started'])
    return run_seed(folder, spec, seed, command, env, spec['timeout_seconds'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--concurrency', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    locks = []
    for path in [BASE / '.exception_rerun.lock'] + [BASE / case / '.lock' for case in CASES]:
        lock = path.open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    plan_path = campaign / 'plan.json'
    if not plan_path.exists():
        jobs = []
        originals = {}
        for case in CASES:
            group = BASE / case
            verify(read(group / 'manifest.json'))
            for seed in SEEDS:
                path = group / str(seed) / 'result.json'
                result = read(path)
                assert result['seed'] == seed
                originals[f'{case}/{seed}'] = sha256(path)
                if result['status'] in EXCEPTIONS:
                    jobs.append({'case': case, 'seed': seed, 'original_status': result['status'],
                                 'original_sha256': sha256(path), 'published': False})
            backup = campaign / 'original' / case
            backup.mkdir(parents=True, exist_ok=True)
            for name in ('manifest.json', 'summary.json'):
                shutil.copy2(group / name, backup / name)
        shutil.copy2(README, campaign / 'README-before.md')
        atomic_json(plan_path, {'jobs': jobs, 'original_hashes': originals,
            'concurrency': args.concurrency, 'attempts_per_exception': 1, 'created': time.time(),
            'authorization': 'User requested replacing ACT/DP exception results with one same-seed rerun.'})
    plan = read(plan_path)
    jobs = plan['jobs']
    if (campaign / 'status.json').exists():
        jobs = read(campaign / 'status.json')['jobs']
    state = 'running'
    def save():
        atomic_json(campaign / 'status.json', {'pid': os.getpid(), 'state': state,
            'updated': time.time(), 'jobs': jobs, 'completed': sum(j.get('published', False) for j in jobs),
            'total': len(jobs), 'concurrency': args.concurrency})
        update_readme(campaign, jobs, state)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, OMNI_KIT_ACCEPT_EULA='YES',
               TORCH_HOME=str(ROOT / '.cache/torch'), OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    save()
    try:
        pending = [job for job in jobs if not job.get('published')]
        pool = ThreadPoolExecutor(max_workers=args.concurrency)
        active = {}
        try:
            while pending or active:
                while pending and len(active) < args.concurrency:
                    if shutil.disk_usage(campaign).free < 8 * 2**30:
                        raise RuntimeError('Disk free below 8 GiB; no further attempts launched')
                    job = pending.pop(0)
                    job.setdefault('started', time.time())
                    active[pool.submit(execute, campaign, job, env)] = job
                    save()
                    print('START/ADOPT', job['case'], job['seed'], flush=True)
                finished, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = active.pop(future)
                    result = future.result()
                    folder = campaign / 'attempts' / job['case'] / str(job['seed'])
                    result['replacement_attempt'] = {'campaign': str(campaign), 'attempt': 2,
                        'original_status': job['original_status'], 'original_sha256': job['original_sha256']}
                    atomic_json(folder / 'result.json', result)
                    publish(campaign, job)
                    job['ended'] = time.time()
                    save()
                    print('DONE', job['case'], job['seed'], result['status'], flush=True)
                    if result['status'] == 'error' and result.get('phase') in ('startup', 'load_policy', 'create_environment'):
                        raise RuntimeError('Infrastructure error; remaining attempts not started')
        finally:
            # Active attempts are allowed to finish before releasing the campaign lock.
            pool.shutdown(wait=True)
        for key, digest in plan['original_hashes'].items():
            if not any(f'{j["case"]}/{j["seed"]}' == key for j in jobs):
                assert sha256(BASE / key / 'result.json') == digest, f'Unselected result changed: {key}'
        for case in CASES:
            summary = summarize(BASE / case, case)
            assert summary['recorded'] == 30 and summary['pending'] == 0
            atomic_json(BASE / case / 'summary.json', summary)
        state = 'completed'
        save()
    except BaseException:
        state = 'halted'
        save()
        raise


if __name__ == '__main__':
    main()
