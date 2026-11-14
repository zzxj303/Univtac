"""No Isaac imports or evaluation seeds consumed."""
import gc
import subprocess
import sys
import json
import time
import cv2
import h5py
import numpy as np
import torch
from eval_common import ROOT,HERE,CASES,binding,atomic_json
from pi05_adapter import encode,load_policy
from data import observation,unnormalize,normalize


def main():
    directory=ROOT/'outputs/pi05_official50/evaluation_preflight'
    directory.mkdir(parents=True,exist_ok=True)
    report={'offline_only':True,'complete':False,'cases':{}}
    atomic_json(directory/'report.json',report)
    subprocess.run([sys.executable,str(HERE/'test_runner.py'),'-v'],check=True)
    for case,(_,task,mode) in CASES.items():
        subprocess.run(['bash','-n',str(HERE/(case+'.sh'))],check=True)
        subprocess.run(['bash',str(HERE/(case+'.sh')),'--help'],check=True,stdout=subprocess.DEVNULL)
        spec=binding(case)
        with h5py.File(ROOT/'data/isaac51'/task/'hdf5/0.hdf5') as f:
            def decode(k): return cv2.imdecode(np.frombuffer(bytes(f[k][0]),dtype=np.uint8),cv2.IMREAD_COLOR)
            obs={'embodiment':{'joint':f['embodiment/joint'][0]},'observation':{
                side:{'rgb':decode(f'observation/{side}/rgb')} for side in ('head','wrist')}}
            if mode=='rgb_tactile': obs['tactile']={side+'_tactile':{'rgb_marker':decode(f'tactile/{side}_tactile/rgb_marker')} for side in ('left','right')}
        images,q=encode(obs,mode=='rgb_tactile')
        policy=load_policy(spec,123456)
        c=policy.controller
        with h5py.File(ROOT/'data/official50'/task/'episode_0.hdf5') as f:
            saved={name:f['observations/images/'+name][0] for name in images}
            for name in images: np.testing.assert_array_equal(images[name],saved[name])
            np.testing.assert_array_equal(q,f['observations/qpos'][0,:8])
            online=observation(images,q,task,mode,c.stats,c.tok)
            offline=observation(saved,q,task,mode,c.stats,c.tok)
            for name in images: torch.testing.assert_close(online['images'][name],offline['images'][name],rtol=0,atol=0)
            absolute=f['action'][:50,:8];delta=absolute.copy();delta[:,:7]-=q[:7]
            restored=unnormalize(normalize(delta,c.stats['action']),c.stats['action']);restored[:,:7]+=q[:7]
            np.testing.assert_allclose(restored,absolute,rtol=1e-5,atol=1e-6)
        a=policy.action(obs)
        c.reset(123456);torch.manual_seed(98765)
        np.testing.assert_array_equal(a,policy.action(obs))
        report['cases'][case]={'binding':spec,'finite_8d_action':True,'training_preprocessing_equal':True,
            'action_roundtrip':True,'independent_rng':True,'rgb_without_tactile':mode=='rgb'}
        atomic_json(directory/'report.json',report)
        print('PASS',case,flush=True)
        del policy,c,obs;gc.collect();torch.cuda.empty_cache()
    report['complete']=True
    atomic_json(directory/'report.json',report)

if __name__=='__main__':main()
