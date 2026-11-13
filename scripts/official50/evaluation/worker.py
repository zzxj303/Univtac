"""Single-seed worker. Only this file launches Isaac; never imported by offline checks."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from common import ROOT, atomic_json, sha256


def main():
    # Parse request without importing Isaac, so boot failures are recorded as results.
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--request', type=Path, required=True)
    request_args, _ = early.parse_known_args()
    request = json.loads(request_args.request.read_text())
    spec, seed, folder = request['spec'], request['seed'], Path(request['folder'])
    if seed not in spec['seeds']:
        raise ValueError('Seed is not in the fixed manifest')
    app = task = None
    actions = 0
    started = time.time()
    result = {'seed': seed, 'status': 'error', 'error': None}
    phase = 'startup'
    def progress():
        atomic_json(folder/'progress.json', {'seed': seed, 'phase': phase, 'actions': actions,
                                             'elapsed_seconds': time.time()-started})
    try:
        progress()
        for key, expected in spec['sha256'].items():
            if sha256(spec['paths'][key]) != expected:
                raise RuntimeError(f'Artifact changed after preflight: {key}')
        sys.path.insert(0, str(ROOT))
        from isaaclab.app import AppLauncher
        parser = argparse.ArgumentParser()
        parser.add_argument('--request', type=Path, required=True)
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        args.enable_cameras = True
        args.num_envs = 1
        args.livestream = 0
        launcher = AppLauncher(args)
        app = launcher.app
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from envs.utils.env_parser import load_task_config, create_task_env
        from policies import load_policy
        torch.set_num_threads(4)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        phase = 'load_policy'
        progress()
        policy = load_policy(spec, seed, args.device)
        config, config_path = load_task_config('clean51', [])
        phase = 'create_environment'
        progress()
        environment = create_task_env(spec['task'], config, config_path.stem, 'eval',
                                       device=args.device, save_dir=folder/'native')
        task = environment.task
        if task.cfg.step_lim != spec['step_limit']:
            raise RuntimeError('Native task action limit changed')
        atomic_json(folder/'resolved_config.json', {
            'task_config': OmegaConf.to_container(config, resolve=True),
            'timing': asdict(environment.timing), 'step_limit': task.cfg.step_lim,
            'reset_time_limit': task.cfg.reset_time_limit,
            'task_config_file': str(config_path), 'seed': seed,
            'policy_spec': spec, 'device': args.device,
            'stage': 'policy control after native reset/pre_move'})
        phase = 'reset'
        progress()
        task.reset(seed=seed, instructions=['Empty'])
        task.mean_steps = task.cfg.step_lim
        phase = 'policy'
        progress()
        def state(obs):
            q = obs['embodiment']['joint']
            return q.detach().cpu().reshape(-1).tolist() if isinstance(q, torch.Tensor) else np.asarray(q).reshape(-1).tolist()
        def write_record(f, row):
            f.write(json.dumps(row, allow_nan=False)+'\n')
            f.flush()
            os.fsync(f.fileno())
        result.update(status='failed', termination='action_limit')
        with (folder/'trajectory.jsonl').open('x') as trace:
            while task.take_action_cnt < task.cfg.step_lim:
                observation = task._get_observations()
                action = policy.action(observation)
                write_record(trace, {'event': 'command', 'index': actions,
                    'state_before': state(observation), 'action': action.tolist()})
                exec_success, eval_success = task.take_action(
                    torch.from_numpy(action).to(task.device).float(), action_type='qpos')
                actions = int(task.take_action_cnt)
                following = task._get_observations()
                early_stop = bool(task.check_early_stop()) if not task.eval_success else False
                write_record(trace, {'event': 'executed', 'actions': actions,
                    'state_after': state(following), 'execution_success': bool(exec_success),
                    'eval_success': bool(task.eval_success), 'early_stop': early_stop})
                progress()
                if task.eval_success:
                    result.update(status='success', termination='native_success')
                    break
                if early_stop:
                    result['termination'] = 'native_early_stop'
                    break
    except BaseException as exc:
        if task is not None:
            actions = int(getattr(task, 'take_action_cnt', actions))
        progress()
        result.update(status='error', error=repr(exc), phase=phase, traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        cleanup_errors = []
        if task is not None:
            try:
                task.clean_cache(result=result['status'])
            except BaseException as exc:
                cleanup_errors.append(f'clean_cache: {exc!r}')
            try:
                task.close()
            except BaseException as exc:
                cleanup_errors.append(f'task.close: {exc!r}')
        result.update(actions=actions, elapsed_seconds=time.time()-started, cleanup_errors=cleanup_errors)
        atomic_json(folder/'worker_result.json', result)
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
