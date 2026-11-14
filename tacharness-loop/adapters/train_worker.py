"""Exact-update, weight-only warm starts using vendored policy code."""
import argparse, json, os, sys, time, math, random, pickle, shutil
from pathlib import Path
from orchestrator.common import ROOT, SOURCE, VENDOR, read, atomic, writable

def prepare_data(task, ids, out):
    # Legacy requests use independent local converted files, never mutable source aliases.
    from adapters.datasets import physical_copy, convert
    target = Path(out) / 'data'
    target.mkdir(parents=True, exist_ok=True)
    for i in ids:
        dst=target / f'episode_{i}.hdf5'
        if dst.exists(): continue
        original=SOURCE / 'data/official50' / task / dst.name
        if original.exists(): physical_copy(original,dst)
        else: convert(SOURCE / 'data/isaac51' / task / 'hdf5' / f'{i}.hdf5',dst)
    return target

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    a = p.parse_args()
    r = read(a.request)
    sys.argv = [sys.argv[0]]
    import torch, numpy as np, yaml
    from torch.utils.data import DataLoader
    g = r['group']
    art = r['artifact']
    out = writable(r['output'])
    out.mkdir(parents=True, exist_ok=True)
    total = r['steps']
    save_at = r.get('save_at_steps') or [total]
    if not total > 0 or save_at != sorted(set(save_at)) or any((not 0 < s <= total for s in save_at)):
        raise ValueError('Invalid update/save schedule')
    save_at = sorted(set(save_at + [total]))
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    split = read(Path(art['train_dir']) / 'split.json') if (Path(art['train_dir']) / 'split.json').exists() else read(SOURCE / 'data/official50' / g['task'] / 'split.json')
    if r.get('train_ids'):
        split = {**split, 'train_ids': r['train_ids']}
    if set(split['train_ids']) & set(split['validation_ids']):
        raise ValueError('Training/validation overlap')
    from adapters.datasets import create_version, validate_version
    version = r.get('dataset_version') or create_version(g, split['train_ids'])
    dataset = validate_version(version, g)
    if dataset['train_ids'] != split['train_ids']:
        version = create_version(g, split['train_ids'], parent=version)
    dataset = validate_version(version, g, split['train_ids'] + split['validation_ids'])
    if dataset['validation_ids'] != split['validation_ids']: raise ValueError('Validation split changed')
    os.environ['TH_DATASET_MANIFEST'] = version
    data_dir = Path(version).parent
    atomic(out / 'split.json', split)
    stats_path = out / Path(art['stats']).name
    shutil.copyfile(art['stats'], stats_path)
    cfg = yaml.safe_load(Path(art['config']).read_text()) if g['policy'] == 'act' else read(art['config'])
    scale = r.get('lr_scale') or 1.0
    started = time.time()
    policy = g['policy']
    save_records = []
    if policy == 'act':
        sys.path.insert(0, str(VENDOR / 'policy/ACT'))
        from act_policy import ACTPolicy
        from utils import TacArenaDataset
        cfg = dict(cfg, num_epochs=6000, device='cuda', seed=42, task_name=g['task'], ckpt_dir=str(out), policy_class='ACT')
        net = ACTPolicy(cfg).cuda()
        state = torch.load(art['checkpoint'], map_location='cpu', weights_only=False)
        net.load_state_dict(state, strict=True)
        del state
        opt = net.configure_optimizers()
        base_lrs = [pg['lr'] * scale for pg in opt.param_groups]
        stats = pickle.load(open(stats_path, 'rb'))
        datasets = [TacArenaDataset(split[k], str(data_dir), cfg['camera_names'], cfg['tactile_names'], stats, cfg['chunk_size']) for k in ['train_ids', 'validation_ids']]
        loaders = [DataLoader(d, batch_size=64, shuffle=i == 0, num_workers=0) for i, d in enumerate(datasets)]

        def loss(batch):
            images, tactile, q, actions, pad = [x.cuda() for x in batch]
            return net(q, images, tactile, actions, pad)['loss']

        def save(step):
            path = out / f'policy_step_{step}.ckpt'
            torch.save(net.state_dict(), path)
            return path
        (out / 'config.yml').write_text(yaml.safe_dump(cfg))
    elif policy == 'dp':
        sys.path.insert(0, str(VENDOR / 'scripts/official50'))
        import dp_adapter
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        class Dataset(dp_adapter.Dataset50):

            def __init__(self, ids, stats):
                super().__init__(g['task'], ids, stats, g['modality'] == 'rgb_tactile')
                self.path = data_dir
        state = torch.load(art['checkpoint'], map_location='cpu', weights_only=False)
        stats = dict(np.load(stats_path))
        names = cfg['camera_names']
        net = torch.nn.ModuleDict({'noise_pred_net': dp_adapter.ConditionalUnet1D(input_dim=8, global_cond_dim=512 * len(names) + 8)})
        for name in names:
            net[name + '_encoder'] = dp_adapter.replace_bn_with_gn(dp_adapter.get_resnet('resnet18'))
        net.load_state_dict(state['state_dict'], strict=True)
        del state
        net.cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-05 * scale, weight_decay=1e-06)
        base_lrs = [1e-05 * scale]
        loaders = [DataLoader(Dataset(split[k], stats), batch_size=bs, shuffle=i == 0, num_workers=0) for i, (k, bs) in enumerate([('train_ids', 8), ('validation_ids', 2)])]
        noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon')

        def loss(batch):
            features = [net[name + '_encoder'](batch[name][:, :1].cuda().flatten(end_dim=1)) for name in names]
            condition = torch.cat(features + [batch['agent_pos'][:, 0].cuda()], dim=-1)
            action = batch['action'].cuda()
            noise = torch.randn_like(action)
            t = torch.randint(0, 100, (len(action),), device='cuda')
            pred = net['noise_pred_net'](noise_scheduler.add_noise(action, noise, t), t, global_cond=condition)
            return torch.nn.functional.mse_loss(pred, noise)

        def save(step):
            path = out / f'policy_step_{step}.ckpt'
            torch.save({'state_dict': net.state_dict(), 'optimizer_steps': step, 'config': cfg, 'stats': stats}, path)
            return path
        atomic(out / 'config.json', cfg)
    else:
        os.environ.update(TH_OUTPUT_ROOT=str(out), TH_SPLIT=str(out / 'split.json'), TH_STATS=str(stats_path), TH_DATA_DIR=str(data_dir))
        sys.path.insert(0, str(VENDOR / 'scripts/pi05_official50'))
        from model import Policy, MasterAdamW
        import model as model_module
        from data import Episodes, loader
        from train import update, validate
        net = Policy(g['modality']).cuda()
        state = torch.load(art['checkpoint'], map_location='cpu', weights_only=False)
        net.load_trainables(state['weights'])
        del state
        opt = MasterAdamW(net)
        base_lrs = [1e-05 * scale, 5e-06 * scale]
        for pg, lr in zip(opt.optimizer.param_groups, base_lrs):
            pg['peak_lr'] = lr
        model_module.lr_multiplier = lambda step: min(step / 50, 1.0)
        ds = Episodes(g['task'], g['modality'])
        val = Episodes(g['task'], g['modality'], 'validation_ids')
        dl, _, _ = loader(ds, 8)
        iterator = iter(dl)
        generator = torch.Generator(device='cuda').manual_seed(42)
        cfg = {**cfg, 'split': split}

        def save(step):
            path = out / f'step_{step}.pt'
            torch.save({'weights': {n: p.detach().cpu().clone() for n, p in net.trainables().items()}, 'step': step, 'config': cfg}, path)
            return path
        atomic(out / 'config.json', cfg)
    from orchestrator.common import require_vendor_imports
    require_vendor_imports(['act_policy', 'utils'] if policy == 'act' else ['dp_adapter'] if policy == 'dp' else ['model', 'data', 'train', 'openpi'])
    if policy != 'pi05':
        iterator = iter(loaders[0])
    selected = [(n, p) for n, p in net.named_parameters() if p.requires_grad][:5]
    before = {n: p.detach().flatten()[:256].cpu().clone() for n, p in selected}
    net.train()
    metrics = []
    for step in range(1, total + 1):
        if shutil.disk_usage(ROOT).free < 8 * 2 ** 30:
            raise RuntimeError('Disk reserve below 8 GiB')
        if policy == 'pi05':
            report = update(net, opt, iterator, generator, step, micro=8, audit=step == 1)
            value = report['loss']
            lrs = [pg['lr'] for pg in opt.optimizer.param_groups]
        else:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loaders[0])
                batch = next(iterator)
            for pg, lr in zip(opt.param_groups, base_lrs):
                pg['lr'] = lr * min(step / 50, 1.0)
            opt.zero_grad()
            value_tensor = loss(batch)
            if not torch.isfinite(value_tensor):
                raise FloatingPointError('Nonfinite training loss')
            value_tensor.backward()
            opt.step()
            value = float(value_tensor.detach())
            lrs = [pg['lr'] for pg in opt.param_groups]
        row = {'step': step, 'optimizer_steps': step, 'train_loss': value, 'lrs': lrs, 'seconds': time.time() - started}
        if step in save_at:
            path = save(step)
            save_records.append({'step': step, 'checkpoint': str(path)})
            net.eval()
            if policy == 'pi05':
                row['validation_loss'] = validate(net, val, 8)['loss']
            else:
                values = []
                with torch.no_grad():
                    for batch in loaders[1]:
                        values.append(float(loss(batch)))
                row['validation_loss'] = sum(values) / len(values)
            net.train()
        from orchestrator.common import append
        append(out / 'metrics.jsonl', row)
        if step == 1 or step % 10 == 0 or step == total:
            print(json.dumps(row), flush=True)
    changed = any((not torch.equal(before[n], p.detach().flatten()[:256].cpu()) for n, p in selected))
    if not changed:
        raise AssertionError('No change in audited trainable parameters')
    saved = torch.load(save_records[-1]['checkpoint'], map_location='cpu', weights_only=False)
    if policy == 'act':
        net.load_state_dict(saved, strict=True)
    elif policy == 'dp':
        net.load_state_dict(saved['state_dict'], strict=True)
    else:
        net.load_trainables(saved['weights'])
    result = dict(dataset_version=version, dataset_sources={str(i):dataset['episodes'][str(i)]['path'] for i in split['train_ids']+split['validation_ids']}, complete=True, actual_updates=total, init_from=art['id'], optimizer_reset=True, parameter_sample_changed=changed, strict_reload=True, checkpoints=save_records, seconds=time.time() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(), vendor_modules={k: str(getattr(v, '__file__', '')) for k, v in sys.modules.items() if k in ['act_policy', 'dp_adapter', 'model', 'data', 'train']})
    atomic(out / 'training_receipt.json', result)
if __name__ == '__main__':
    main()
