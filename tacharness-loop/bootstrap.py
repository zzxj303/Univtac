"""Snapshot current working files without invoking git or touching source trees."""
import json, shutil, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SOURCES = {'/root/autodl-tmp/gujie/UniVTAC': ['envs', 'assets', 'task_config', 'policy/ACT', 'policy/ViTAL/diffusion', 'scripts/official50', 'scripts/pi05_official50'], '/root/autodl-tmp/gujie/OpenPI-RoboLab': ['src', 'packages/openpi-client/src']}
rows = []
for source, parts in SOURCES.items():
    source = Path(source)
    for part in parts:
        for p in sorted((source / part).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc', '.log'):
                continue
            dest = ROOT / 'vendor' / source.name / p.relative_to(source)
            if dest.exists():
                raise RuntimeError(f'Refusing to overwrite snapshot: {dest}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
            rows.append(dict(source=str(p), destination=str(dest.relative_to(ROOT)), copied_at=time.time(), bytes=p.stat().st_size))
(ROOT / 'vendor/MANIFEST.json').write_text(json.dumps({'version': 1, 'files': rows}, indent=2))
print('copied', len(rows), 'files')
