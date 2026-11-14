"""Read-only source catalog and explicit, immutable demonstration versions."""
import json, os, shutil, time, uuid
from pathlib import Path
import h5py
import numpy as np
from orchestrator.common import ROOT, SOURCE, atomic, read, writable

CHANNELS = {'cam_high': 'observation/head/rgb', 'cam_wrist': 'observation/wrist/rgb',
            'tac_left': 'tactile/left_tactile/rgb_marker', 'tac_right': 'tactile/right_tactile/rgb_marker'}

def catalog(group):
    task = group['task']
    split = read(SOURCE / 'data/official50' / task / 'split.json')
    metadata = read(SOURCE / 'data/isaac51' / task / 'metadata.json')
    rows = []
    for i in range(100):
        meta = metadata.get(str(i))
        if meta is None:
            continue
        seeds = {meta.get('seed'), meta.get('source_seed')}
        eligible = i not in split['validation_ids'] and None not in seeds and not seeds.intersection(group['feedback_seeds'] + group['test_seeds'])
        raw = SOURCE / 'data/isaac51' / task / 'hdf5' / f'{i}.hdf5'
        prepared = SOURCE / 'data/official50' / task / f'episode_{i}.hdf5'
        rows.append({'demo_id': i, 'source_seed': meta.get('source_seed'), 'seed': meta.get('seed'),
                     'partition': 'validation' if i in split['validation_ids'] else 'train' if i in split['train_ids'] else 'additional',
                     'eligible_for_repair': eligible and raw.is_file(), 'raw_path': str(raw),
                     'training_path': str(prepared) if prepared.is_file() else None,
                     'metadata_path': str(SOURCE / 'data/isaac51' / task / 'metadata.json'), 'metadata_field': '/' + str(i)})
    return rows

def source_files(group):
    files = {str(SOURCE / 'data/official50' / group['task'] / 'split.json')}
    for r in catalog(group):
        # Validation trajectories may be inspected but never edited; evaluation seeds are excluded.
        if r['seed'] in group['feedback_seeds'] + group['test_seeds'] or r['source_seed'] in group['feedback_seeds'] + group['test_seeds']:
            continue
        for k in ('raw_path', 'training_path', 'metadata_path'):
            if r.get(k) and Path(r[k]).is_file():
                files.add(r[k])
    return sorted(files)

def physical_copy(source, destination):
    """Copy bytes, never link a mutable destination to original storage."""
    src, dst = Path(source), writable(destination)
    if dst.exists():
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < src.stat().st_size + 2 * 2**30:
        raise RuntimeError('Insufficient space for independent demonstration copy')
    shutil.copyfile(src, dst)
    if dst.is_symlink() or dst.stat().st_nlink != 1 or os.path.samefile(src, dst):
        raise RuntimeError('Demonstration destination is not an independent file')
    if dst.suffix in ('.h5', '.hdf5'):
        with h5py.File(dst, 'r') as f:
            def inspect(g):
                for name in g:
                    link = g.get(name, getlink=True)
                    if not isinstance(link, h5py.HardLink):
                        raise ValueError('HDF5 external/soft links are not supported')
                    obj = g[name]
                    if isinstance(obj, h5py.Group): inspect(obj)
                    elif obj.is_virtual or obj.external: raise ValueError('External/virtual HDF5 storage is not supported')
            inspect(f)
    return {'source': str(src), 'destination': str(dst), 'bytes': dst.stat().st_size, 'copied_at': time.time()}

def convert(raw, destination):
    """Use the same next-observed-qpos labels and BGR resizing as the baseline."""
    import cv2
    dst = writable(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw, 'r') as f, h5py.File(dst, 'x') as w:
        q = f['embodiment/joint'][:, :8].astype('float32')
        if len(q) < 2 or not np.isfinite(q).all(): raise ValueError('Invalid observed trajectory')
        w.create_dataset('action', data=q[1:]); w.create_dataset('observations/qpos', data=q[:-1])
        for name, key in CHANNELS.items():
            if len(f[key]) != len(q): raise ValueError('Observation/action alignment mismatch')
            ds = w.create_dataset('observations/images/' + name, (len(q)-1,256,256,3), dtype='uint8', compression='lzf', chunks=(1,256,256,3))
            for t in range(len(q)-1):
                buf = f[key][t]
                img = cv2.imdecode(np.frombuffer(bytes(buf), dtype='uint8'), cv2.IMREAD_COLOR) if np.asarray(buf).ndim < 3 else np.asarray(buf)
                if img is None: raise ValueError('Cannot decode recorded image')
                ds[t] = cv2.resize(img, (256,256), interpolation=cv2.INTER_LINEAR)
    return dst

def manifest_path(version):
    p = writable(version)
    if p.name != 'manifest.json' or not p.is_relative_to(ROOT / 'runs/datasets'):
        raise ValueError('Dataset version must be a registered manifest under runs/datasets')
    return p

def validate_version(version, group, ids=None):
    p = manifest_path(version); m = read(p)
    if m['task'] != group['task'] or m['group'] != group['id'] or not m.get('sealed'):
        raise ValueError('Dataset version identity/sealing mismatch')
    if m.get('development_only') and not group.get('_technical_acceptance'):
        raise ValueError('Synthetic acceptance datasets cannot enter a formal experiment')
    for i in ids or m['train_ids']:
        entry = m['episodes'].get(str(i))
        if entry is None or not Path(entry['path']).is_file(): raise ValueError('Dataset episode missing')
        if entry.get('modified'):
            owned = writable(entry['path'])
            if owned.is_symlink() or owned.stat().st_nlink != 1: raise ValueError('Modified episode must be independent')
            proof = read(entry['acceptance'])
            if not proof.get('accepted') or proof['demo_id'] != i: raise ValueError('Repair has not passed independent verification')
    return m

def create_version(group, ids, replacements=None, parent=None):
    replacements = replacements or {}
    split = read(SOURCE / 'data/official50' / group['task'] / 'split.json')
    from adapters.registry import get_adapter
    errors = get_adapter(group).validate_training_data(group, ids)
    if errors: raise ValueError('; '.join(errors))
    previous = validate_version(parent,group) if parent else None
    folder = ROOT / 'runs/datasets' / group['id'] / uuid.uuid4().hex
    folder.mkdir(parents=True)
    entries = {}; provenance = []
    rows = {r['demo_id']:r for r in catalog(group)}
    for i in sorted(set(ids + split['validation_ids'])):
        if i in replacements:
            if i in split['validation_ids']: raise ValueError('Validation modification forbidden')
            proof_path = Path(replacements[i]); proof = read(proof_path)
            if not proof.get('accepted') or proof['demo_id'] != i or (proof['group'] != group['id'] and not (group.get('_technical_acceptance') and proof.get('scope') == 'synthetic technical acceptance only' and proof.get('task') == group['task'])):
                raise ValueError('Invalid accepted repair')
            raw = folder / f'raw_{i}.hdf5'
            provenance.append(physical_copy(proof['recording'], raw))
            dst = convert(raw,folder / f'episode_{i}.hdf5')
            entries[str(i)] = {'path':str(dst),'modified':True,'acceptance':str(proof_path),'source_seed':rows[i]['source_seed']}
        elif previous and str(i) in previous['episodes']:
            entries[str(i)] = previous['episodes'][str(i)]
        elif rows[i]['training_path']:
            entries[str(i)] = {'path':rows[i]['training_path'],'modified':False,'source_seed':rows[i]['source_seed']}
        else:
            raw = folder / f'raw_{i}.hdf5'; provenance.append(physical_copy(rows[i]['raw_path'],raw))
            dst = convert(raw,folder / f'episode_{i}.hdf5')
            entries[str(i)] = {'path':str(dst),'modified':False,'source_seed':rows[i]['source_seed']}
    development = bool(previous and previous.get('development_only')) or any(read(v).get('scope') == 'synthetic technical acceptance only' for v in replacements.values())
    if development and not group.get('_technical_acceptance'): raise ValueError('Synthetic data requires explicit trusted technical-acceptance mode')
    m = {'version':1,'development_only':development,'group':group['id'],'task':group['task'],'parent':parent,'train_ids':ids,'validation_ids':split['validation_ids'],
         'episodes':entries,'copies':provenance,'sealed':True,'created_at':time.time(),
         'unchanged_sources':'Explicit read-only paths; no links or writable aliases'}
    atomic(folder/'manifest.json',m,exclusive=True)
    return str(folder/'manifest.json')

def resolve_episode(directory, episode_id):
    manifest = os.environ.get('TH_DATASET_MANIFEST')
    if manifest:
        return Path(read(manifest)['episodes'][str(int(episode_id))]['path'])
    return Path(directory) / f'episode_{episode_id}.hdf5'
