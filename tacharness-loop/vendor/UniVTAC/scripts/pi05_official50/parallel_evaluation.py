"""Parallel supervisor; does not modify immutable evaluation/worker sources."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'scripts/pi05_official50/evaluation'
sys.path.insert(0,str(HERE))
from eval_common import CASES,atomic_json,summarize
from runner import process_identity,terminate_group
from campaign import resources
OUT=ROOT/'outputs/pi05_official50'


def alive(pid,identity):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return fields[0]!='Z' and fields[19]==identity
    except FileNotFoundError:return False


def active_phase(case):
    folder=OUT/'evaluation'/case
    pending=[p for p in folder.glob('*/started.json') if not (p.parent/'result.json').exists()]
    if not pending:return 'between_episodes'
    p=pending[-1].parent/'progress.json'
    return json.loads(p.read_text()).get('phase','startup') if p.exists() else 'startup'


def stop_runner(info):
    if not alive(info['pid'],info['identity']):return
    os.kill(info['pid'],signal.SIGTERM)
    for _ in range(100):
        if not alive(info['pid'],info['identity']):return
        time.sleep(.1)
    # Last resort: only this runner's recorded worker process group.
    for p in (OUT/'evaluation'/info['case']).glob('*/started.json'):
        if (p.parent/'result.json').exists():continue
        row=json.loads(p.read_text())
        if row.get('pid') and alive(row['pid'],row.get('process_identity')):terminate_group(row['pid'])
    if alive(info['pid'],info['identity']):os.kill(info['pid'],signal.SIGKILL)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--concurrency',type=int,choices=(1,2,3,4),default=2)
    parser.add_argument('--adopt',type=Path)
    args=parser.parse_args()
    folder=OUT/'evaluation_launch';folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active={};children={};completed=[]
        if args.adopt:
            adoption=json.loads(args.adopt.read_text())
            for row in adoption['runners']:
                if row['case'] not in CASES:raise ValueError('Unknown adopted case')
                active[row['case']]=row
        pending=[case for case in CASES if case not in active]
        state={'pid':os.getpid(),'gpu':args.gpu,'state':'formal','concurrency':args.concurrency,
               'cases':list(CASES),'integration_passed':False,'operator_requested_unvalidated_run':True,
               'stop_on_timeout':False,'started':time.time(),
               'supervisor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        def save(**kw):
            state.update(kw);state.update(active_cases=list(active),active=active,completed_cases=completed,pending_cases=pending)
            atomic_json(folder/'status.json',state)
        def interrupt(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
        save()
        try:
            while active or pending:
                for case,info in list(active.items()):
                    if alive(info['pid'],info['identity']):continue
                    child=children.pop(case,None)
                    code=child.wait() if child else None
                    summary=summarize(OUT/'evaluation'/case,case)
                    if code not in (None,0) or not summary['complete']:
                        raise RuntimeError(f'{case} stopped: exit={code}, recorded={summary["recorded"]}; stopping all evaluation per operator rule')
                    completed.append(case);del active[case]
                usage=resources(args.gpu)
                # Add a slot after existing workers enter policy, avoiding startup bursts.
                ready=not active or all(active_phase(case)=='policy' for case in active)
                if pending and len(active)<args.concurrency and ready and usage['gpu_free_mib']>=24576 and usage['ram_free_gib']>=24 and usage['disk_free_gib']>=8:
                    case=pending.pop(0)
                    with (folder/(case+'.log')).open('ab',buffering=0) as log:
                        child=subprocess.Popen(['bash',str(HERE/(case+'.sh')),'--gpu',args.gpu,
                            '--allow-unvalidated-integration'],cwd=ROOT,
                            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    children[case]=child
                    active[case]={'case':case,'pid':child.pid,'identity':process_identity(child.pid),'adopted':False}
                save(resources=usage,phases={case:active_phase(case) for case in active})
                if active or pending:time.sleep(3)
            save(state='completed')
        except BaseException as exc:
            save(state='stopping',error=repr(exc))
            for info in active.values():stop_runner(info)
            for child in children.values():child.wait()
            save(state='stopped');raise

if __name__=='__main__':main()
