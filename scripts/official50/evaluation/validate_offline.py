"""Offline only: validate twelve trained policies without importing Isaac or launching rollouts."""
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np
import torch
from common import ROOT, HERE, CASES, binding, atomic_json
from policies import load_policy, encode, RGB_NAMES, TAC_NAMES, RGB_TRANSFORM, TAC_TRANSFORM


def observation(task, tactile):
    # A previously recorded training observation; no simulation or evaluation seed consumed.
    with h5py.File(ROOT/'data/isaac51'/task/'hdf5/0.hdf5') as f:
        def decode(key):
            return cv2.imdecode(np.frombuffer(bytes(f[key][0]),dtype=np.uint8),cv2.IMREAD_COLOR)
        obs={'embodiment':{'joint':f['embodiment/joint'][0]},'observation':{
            side:{'rgb':decode(f'observation/{side}/rgb')} for side in ['head','wrist']}}
        if tactile:
            obs['tactile']={f'{side}_tactile':{'rgb_marker':decode(f'tactile/{side}_tactile/rgb_marker')} for side in ['left','right']}
    return obs


def main():
    torch.set_num_threads(4)
    report={'offline_only':True,'formal_rollouts_started':0,'cases':{}}
    directory=ROOT/'outputs/official50/evaluation_preflight'
    directory.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(HERE/'test_runner.py'),'-v'],check=True)
    for case,(algorithm,task,mode) in CASES.items():
        started=time.time()
        subprocess.run(['bash','-n',str(HERE/(case+'.sh'))],check=True)
        help_result=subprocess.run(['bash',str(HERE/(case+'.sh')),'--help'],capture_output=True,text=True,check=True)
        assert '--gpu' in help_result.stdout and '--output-root' in help_result.stdout
        rejected=subprocess.run(['bash',str(HERE/(case+'.sh')),'--ca',case],capture_output=True)
        assert rejected.returncode != 0
        spec=binding(case)
        obs=observation(task,mode=='rgb_tactile')
        if mode=='rgb': assert 'tactile' not in obs
        encoded=encode(obs,mode=='rgb_tactile')
        with h5py.File(ROOT/'data/official50'/task/'episode_0.hdf5') as f:
            for name in RGB_NAMES+(TAC_NAMES if mode=='rgb_tactile' else []):
                transform=RGB_TRANSFORM if name in RGB_NAMES else TAC_TRANSFORM
                torch.testing.assert_close(encoded[name],transform(f['observations/images/'+name][0]),rtol=0,atol=0)
        policy=load_policy(spec,123456,'cuda:0')  # offline sampling seed, not a formal environment seed
        action=policy.action(obs)
        assert action.shape==(8,) and np.isfinite(action).all()
        details={'checkpoint_sha256':spec['sha256']['checkpoint'],'stats_sha256':spec['sha256']['stats'],
                 'seed_overlap':spec['seed_audit']['overlap'],'action_shape':list(action.shape),
                 'finite_action':True,'preprocessing_matches_saved_training_frame':True,
                 'rgb_without_tactile_field':mode=='rgb','seconds':time.time()-started}
        if algorithm=='dp':
            assert policy.last_denoising_steps==10
            details['denoising_steps']=policy.last_denoising_steps
            policy.generator.manual_seed(123456)
            torch.manual_seed(98765)  # global RNG must not change the DP sampling result
            np.testing.assert_array_equal(action,policy.action(obs))
            details['independent_rng_verified']=True
        else:
            assert policy.model.t==1
            second=policy.action(obs)
            assert np.isfinite(second).all() and policy.model.t==2
            details['temporal_aggregation_checked']=True
        report['cases'][case]=details
        atomic_json(directory/'report.json',report)
        print('PASS',case,flush=True)
        del policy,encoded,obs
        gc.collect();torch.cuda.empty_cache()
    assert len(report['cases'])==12
    print('OFFLINE_VALIDATION_COMPLETE; no formal simulation episodes launched',flush=True)


if __name__=='__main__':main()
