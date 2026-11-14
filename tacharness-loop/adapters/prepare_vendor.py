"""One-time initial migration. The delivered vendor snapshot includes subsequent fixes.

Do not rerun this migration on the adapted snapshot; it is retained as provenance.
"""
from pathlib import Path
import json
from orchestrator.common import ROOT, SOURCE, VENDOR

if "spec['sha256']" not in (VENDOR / 'scripts/official50/evaluation/worker.py').read_text():
    raise RuntimeError('Snapshot already adapted; use the delivered vendor sources')

def edit(rel, fn):
    p = VENDOR / rel
    s = p.read_text()
    p.write_text(fn(s))
edit('policy/ACT/detr/models/backbone.py', lambda s: s.replace('pretrained=is_main_process()', 'pretrained=False'))
edit('scripts/official50/evaluation/policies.py', lambda s: s.replace("str(ROOT / '.cache/torch')", "str(__import__('pathlib').Path(__import__('os').environ['TORCH_HOME']))").replace("checkpoint_name='policy_best.ckpt'", "checkpoint_name=__import__('pathlib').Path(spec['paths']['checkpoint']).name"))
common = 'from pathlib import Path\nfrom orchestrator.common import atomic as atomic_json, VENDOR as ROOT\n'
(VENDOR / 'scripts/official50/evaluation/common.py').write_text(common)
(VENDOR / 'scripts/pi05_official50/evaluation/eval_common.py').write_text(common)
for rel in ['scripts/official50/evaluation/worker.py', 'scripts/pi05_official50/evaluation/worker.py']:

    def patch(s):
        s = s.replace('ROOT, atomic_json, sha256', 'ROOT, atomic_json')
        start = s.index("        for key, expected in spec['sha256'].items():")
        end = s.index('        sys.path.insert(0, str(ROOT))', start)
        s = s[:start] + s[end:]
        s = s.replace("site.addsitedir(str(ROOT/'.venv/lib/python3.11/site-packages'))", f"site.addsitedir({str(SOURCE / '.venv/lib/python3.11/site-packages')!r})\n        sys.path.insert(0, str(ROOT.parent/'OpenPI-RoboLab/src'))\n        sys.path.insert(0, str(ROOT))")
        s = s.replace('launcher = AppLauncher(args)', "args.kit_args = (getattr(args, 'kit_args', '') + ' --/log/file=' + str(folder/'kit.log') + ' --/app/userConfigPath=' + str(folder/'kit-user.config.json') + ' --/app/localCachePath=' + os.environ['XDG_CACHE_HOME'] + ' --/app/cachePath=' + os.environ['XDG_CACHE_HOME'])\n        launcher = AppLauncher(args)")
        return s
    edit(rel, patch)
p = VENDOR / 'scripts/pi05_official50/common.py'
s = p.read_text()
s = s.replace('ROOT=Path(__file__).resolve().parents[2]', f'ROOT=Path({str(SOURCE)!r})')
s = s.replace("OUT=ROOT/'outputs/pi05_official50'", "OUT=Path(os.environ['TH_OUTPUT_ROOT'])")
s = s.replace("BASE=OUT/'base/model.safetensors'", f"BASE=Path({str(SOURCE / 'outputs/pi05_official50/base/model.safetensors')!r})")
a = s.index('def sha256(path):')
b = s.index('def atomic_json(', a)
s = s[:a] + s[b:]
p.write_text(s)
p = VENDOR / 'scripts/pi05_official50/data.py'
s = p.read_text().replace('atomic_json,sha256,seed_worker', 'atomic_json,seed_worker')
s = 'import os\n' + s
start = s.index('def split_for(task):')
end = s.index('def targets(', start)
s = s[:start] + "def split_for(task):\n return json.loads(open(os.environ['TH_SPLIT']).read())\n\n" + s[end:]
start = s.index('def stats_for(task):')
end = s.index('def normalize(', start)
s = s[:start] + "def stats_for(task):\n return json.loads(open(os.environ['TH_STATS']).read())\n\n" + s[end:]
s = s.replace("self.directory=ROOT/'data/official50'/task", "self.directory=__import__('pathlib').Path(os.environ['TH_DATA_DIR'])")
p.write_text(s)
p = VENDOR / 'scripts/pi05_official50/inference.py'
s = p.read_text().replace('OUT,BASE,sha256', 'OUT,BASE')
s = '\n'.join((line for line in s.splitlines() if 'assert sha256(' not in line)) + '\n'
p.write_text(s)
for name in ['data']:
    p = VENDOR / name
    if not p.exists():
        p.symlink_to(SOURCE / name, target_is_directory=True)
print('vendor execution sources adapted')
