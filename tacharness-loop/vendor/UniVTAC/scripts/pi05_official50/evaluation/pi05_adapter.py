"""Simulation observation adapter; all model preprocessing stays in training code."""
import sys
import json
import cv2
import numpy as np
import torch
from eval_common import ROOT
sys.path.insert(0, str(ROOT/'scripts/pi05_official50'))
from inference import Pi05Controller


def encode(obs, tactile):
    def array(x):
        return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
    q = array(obs['embodiment']['joint']).reshape(-1)[:8].astype(np.float32)
    if q.shape != (8,) or not np.isfinite(q).all(): raise ValueError('Invalid qpos')
    def image(x):
        x=array(x)
        if x.ndim!=3 or x.shape[-1]!=3 or x.dtype!=np.uint8: raise ValueError('Invalid image')
        return cv2.resize(x,(256,256),interpolation=cv2.INTER_LINEAR)
    images={name:image(obs['observation'][side]['rgb']) for name,side in
            [('cam_high','head'),('cam_wrist','wrist')]}
    if tactile:
        for side in ('left','right'):
            source=obs['tactile'];key=side+'_tactile' if side+'_tactile' in source else side+'_gsmini'
            images['tac_'+side]=image(source[key]['rgb_marker'])
    return images,q


class Pi05Adapter:
    def __init__(self,spec,seed,device):
        self.controller=Pi05Controller(spec['paths']['checkpoint'],device)
        self.tactile=spec['modality']=='rgb_tactile'
        config=json.load(open(spec['paths']['training_config']))
        if self.controller.config!=config: raise ValueError('Checkpoint/config mismatch')
        if config['task']!=spec['task'] or config['mode']!=spec['modality']:
            raise ValueError('Task/modality mismatch')
        self.controller.reset(seed)
    def action(self,obs):
        images,q=encode(obs,self.tactile)
        a=self.controller.predict(images,q)[0]
        if a.shape!=(8,) or not np.isfinite(a).all(): raise ValueError('Invalid action')
        return a


def load_policy(spec,seed,device='cuda:0'):
    return Pi05Adapter(spec,seed,device)
