"""Checkpoint adapters; no simulator import or environment side effects."""
import importlib.util
import json
import os
import pickle
import sys

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms
from common import ROOT

RGB_NAMES = ['cam_high', 'cam_wrist']
TAC_NAMES = ['tac_left', 'tac_right']
RGB_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
TAC_TRANSFORM = transforms.ToTensor()


def numpy_image(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise ValueError(f'Expected HWC uint8 image, got {value.shape}/{value.dtype}')
    # Same OpenCV resize and channel order as training data preparation.
    return cv2.resize(value, (256, 256), interpolation=cv2.INTER_LINEAR)


def encode(observation, tactile):
    qpos = observation['embodiment']['joint']
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()
    qpos = np.asarray(qpos).reshape(-1)[:8].astype(np.float32)
    if qpos.shape != (8,) or not np.isfinite(qpos).all():
        raise ValueError('Invalid 8-dimensional robot state')
    result = {'qpos': qpos}
    for name, key in zip(RGB_NAMES, ('head', 'wrist')):
        result[name] = RGB_TRANSFORM(numpy_image(observation['observation'][key]['rgb']))
    if tactile:
        for name, side in zip(TAC_NAMES, ('left', 'right')):
            source = observation['tactile']
            key = f'{side}_tactile' if f'{side}_tactile' in source else f'{side}_gsmini'
            result[name] = TAC_TRANSFORM(numpy_image(source[key]['rgb_marker']))
    return result


def checked_action(action):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (8,) or not np.isfinite(action).all():
        raise ValueError('Policy returned invalid action')
    return action


class ACTAdapter:
    def __init__(self, spec, seed, device='cuda:0'):
        os.environ['TORCH_HOME'] = str(ROOT / '.cache/torch')
        sys.path.insert(0, str(ROOT / 'policy/ACT'))
        from act_policy import ACT
        cfg = yaml.safe_load(open(spec['paths']['training_config']))
        self.tactile = spec['modality'] == 'rgb_tactile'
        assert cfg['camera_names'] == RGB_NAMES
        assert cfg['tactile_names'] == (TAC_NAMES if self.tactile else [])
        with open(spec['paths']['stats'], 'rb') as f:
            stats = pickle.load(f)
        for key in ('qpos_mean', 'qpos_std', 'action_mean', 'action_std'):
            assert np.asarray(stats[key]).shape == (8,) and np.isfinite(stats[key]).all()
        assert (stats['qpos_std'] > 0).all() and (stats['action_std'] > 0).all()
        cfg.update(device=device, task_name=spec['task'], seed=seed, num_epochs=6000,
                   ckpt_dir=str(__import__('pathlib').Path(spec['paths']['checkpoint']).parent),
                   checkpoint_name='policy_best.ckpt')
        self.model = ACT(cfg)
        assert self.model.temporal_agg and self.model.num_queries == 50
        self.model.reset()

    def action(self, observation):
        return checked_action(self.model.get_action(encode(observation, self.tactile)))


class DPAdapter:
    def __init__(self, spec, seed, device='cuda:0'):
        self.device = torch.device(device)
        self.tactile = spec['modality'] == 'rgb_tactile'
        self.names = RGB_NAMES + (TAC_NAMES if self.tactile else [])
        module_spec = importlib.util.spec_from_file_location('official50_dp_network',
            ROOT / 'policy/ViTAL/diffusion/network.py')
        network = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(network)
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        saved = torch.load(spec['paths']['checkpoint'], map_location='cpu', weights_only=False)
        cfg = json.load(open(spec['paths']['training_config']))
        for key in ('task', 'modality', 'camera_names', 'action_dim', 'state_dim', 'pred_horizon', 'obs_horizon'):
            assert saved['config'][key] == cfg[key], key
        assert cfg['task'] == spec['task'] and cfg['modality'] == spec['modality']
        assert cfg['camera_names'] == self.names and cfg['action_dim'] == cfg['state_dim'] == 8
        assert cfg['pred_horizon'] == 20 and cfg['obs_horizon'] == 1
        self.nets = torch.nn.ModuleDict({'noise_pred_net': network.ConditionalUnet1D(
            input_dim=8, global_cond_dim=512*len(self.names)+8)})
        for name in self.names:
            self.nets[name+'_encoder'] = network.replace_bn_with_gn(network.get_resnet('resnet18'))
        self.nets.load_state_dict(saved['state_dict'], strict=True)
        self.nets.to(self.device).eval()
        with np.load(spec['paths']['stats']) as stats:
            self.stats = {k: stats[k].copy() for k in stats.files}
        for key in ('qpos_min', 'qpos_max', 'action_min', 'action_max'):
            assert self.stats[key].shape == (8,) and np.isfinite(self.stats[key]).all()
            np.testing.assert_array_equal(self.stats[key], saved['stats'][key])
        self.scheduler = DDPMScheduler(num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon')
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.last_denoising_steps = 0

    @torch.inference_mode()
    def action(self, observation):
        obs = encode(observation, self.tactile)
        features = [self.nets[name+'_encoder'](obs[name][None].to(self.device)) for name in self.names]
        s = self.stats
        qpos = 2*(obs['qpos']-s['qpos_min'])/np.maximum(s['qpos_max']-s['qpos_min'], 1e-6)-1
        condition = torch.cat(features+[torch.from_numpy(qpos)[None].to(self.device)], dim=-1)
        action = torch.randn((1, 20, 8), device=self.device, generator=self.generator)
        self.scheduler.set_timesteps(10)
        self.last_denoising_steps = 0
        for timestep in self.scheduler.timesteps:
            predicted = self.nets['noise_pred_net'](action, timestep, global_cond=condition)
            action = self.scheduler.step(predicted, timestep, action, generator=self.generator).prev_sample
            self.last_denoising_steps += 1
        normalized = action[0, 0].cpu().numpy()
        # Invert exactly the training normalization, including its zero-range clamp.
        result = (normalized+1)/2*np.maximum(s['action_max']-s['action_min'], 1e-6)+s['action_min']
        return checked_action(result)


def load_policy(spec, seed, device='cuda:0'):
    return (ACTAdapter if spec['policy'] == 'act' else DPAdapter)(spec, seed, device)
