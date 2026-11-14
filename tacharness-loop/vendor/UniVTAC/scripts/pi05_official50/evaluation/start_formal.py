"""Operator-requested serial six-script run; integration failure remains disclosed."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from eval_common import ROOT,HERE,CASES,atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',default='0')
    args=parser.parse_args()
    folder=ROOT/'outputs/pi05_official50/evaluation_launch';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'pid':os.getpid(),'gpu':args.gpu,'state':'formal','concurrency':1,
               'cases':list(CASES),'completed_cases':[], 'pending_cases':list(CASES),
               'integration_passed':False,'operator_requested_unvalidated_run':True,
               'stop_on_timeout':False,'started':time.time()}
        def save(**kw): state.update(kw);atomic_json(folder/'status.json',state)
        def interrupt(signum,frame): raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
        save()
        try:
            for case in CASES:
                save(case=case)
                with (folder/(case+'.log')).open('ab',buffering=0) as log:
                    process=subprocess.Popen(['bash',str(HERE/(case+'.sh')),'--gpu',args.gpu,
                        '--allow-unvalidated-integration'],cwd=ROOT,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    try: code=process.wait()
                    except BaseException:
                        process.terminate()
                        try: process.wait(timeout=15)
                        except subprocess.TimeoutExpired: process.kill();process.wait()
                        raise
                if code: raise RuntimeError(f'{case} stopped with exit {code}; no more scripts launched. See group log/result.')
                state['completed_cases'].append(case);state['pending_cases'].remove(case);save()
            save(state='completed',case=None)
        except BaseException as exc:
            save(state='stopped',error=repr(exc));raise

if __name__=='__main__':main()
