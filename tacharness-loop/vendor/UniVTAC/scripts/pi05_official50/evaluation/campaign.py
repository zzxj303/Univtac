"""Serial integration acceptance, then the fixed four-case formal campaign."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from eval_common import ROOT,HERE,CASES,SEEDS,atomic_json,binding
from runner import run_seed

OUT=ROOT/'outputs/pi05_official50'

def resources(gpu):
    free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
    return {'gpu_free_mib':free,'disk_free_gib':shutil.disk_usage(ROOT).free/2**30,
            'ram_free_gib':mem['MemAvailable']/2**20}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--integration-attempt',default='initial',help='Separate diagnostic attempt; never repeats formal seeds')
    args=parser.parse_args()
    if not args.integration_attempt.replace('_','').replace('-','').isalnum(): parser.error('Invalid integration attempt name')
    launch=OUT/'evaluation_launch';launch.mkdir(parents=True,exist_ok=True)
    with (launch/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'pid':os.getpid(),'gpu':args.gpu,'state':'preflight','concurrency':1,'cases':list(CASES)}
        def save(**kw): state.update(kw);atomic_json(launch/'status.json',state)
        def wait_resources():
            while True:
                r=resources(args.gpu);save(resources=r)
                if r['gpu_free_mib']>=24576 and r['disk_free_gib']>=8 and r['ram_free_gib']>=24: return
                save(state='waiting_resources');time.sleep(30)
        def interrupt(signum,frame): raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
        save()
        try:
            pre=json.loads((OUT/'evaluation_preflight/report.json').read_text())
            assert pre['complete'] and set(pre['cases'])==set(CASES),'Offline validation required'
            specs={case:binding(case) for case in CASES}
            for case,spec in specs.items():
                assert spec['sha256']==pre['cases'][case]['binding']['sha256'],'Offline artifacts changed'
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,OMNI_KIT_ACCEPT_EULA='YES',
                JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',TOKENIZERS_PARALLELISM='false',
                OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            # Pick deterministic, paired diagnostic seeds outside every excluded/formal set.
            excluded=set(SEEDS)
            for spec in specs.values(): excluded.update(spec['seed_audit']['excluded_seed_and_source_seed'])
            seed=30001
            while seed in excluded: seed+=1
            for case,spec in specs.items():
                wait_resources();save(state='integration',case=case)
                folder=OUT/'evaluation_integration'/args.integration_attempt/case/str(seed)
                diagnostic=dict(spec,seeds=[seed],evaluation_kind='integration')
                folder.parent.mkdir(parents=True,exist_ok=True)
                manifest=folder.parent/'manifest.json'
                if manifest.exists(): assert json.loads(manifest.read_text())==diagnostic,'Integration source changed'
                else: atomic_json(manifest,diagnostic)
                command=[sys.executable,'-u',str(HERE/'worker.py'),'--request',str(folder/'request.json'),'--headless','--device','cuda:0']
                result=run_seed(folder,diagnostic,seed,command,env,1800)
                if result['status'] not in ('success','failed') or result.get('worker',{}).get('cleanup_errors'):
                    raise RuntimeError(f'Integration failed: {case}: {result}')
                print('INTEGRATION_PASS',case,result['status'],flush=True)
            atomic_json(OUT/'evaluation_integration/acceptance.json',{'complete':True,'cases':list(CASES),'seed':seed,'bindings':specs,'attempt':args.integration_attempt})
            for case in CASES:
                wait_resources();save(state='formal',case=case)
                command=[sys.executable,'-u',str(HERE/'runner.py'),'--case',case,'--gpu',args.gpu]
                with (launch/(case+'.log')).open('ab',buffering=0) as log:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    try:
                        code=child.wait()
                    except BaseException:
                        child.terminate()
                        try: child.wait(timeout=15)
                        except subprocess.TimeoutExpired: child.kill();child.wait()
                        raise
                if code: raise RuntimeError(f'Formal group exited {code}: {case}')
            save(state='completed',case=None)
        except BaseException as exc:
            save(state='stopped',error=repr(exc));raise

if __name__=='__main__':main()
