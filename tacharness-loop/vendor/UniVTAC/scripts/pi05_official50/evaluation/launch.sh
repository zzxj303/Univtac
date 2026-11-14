#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../.." && pwd)"
exec "$ROOT/.venv-pi05/bin/python" - "$ROOT" "$@" <<'PY'
import pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);out=root/'outputs/pi05_official50/evaluation_launch'
out.mkdir(parents=True,exist_ok=True)
with (out/'campaign.log').open('ab',buffering=0) as log:
    process=subprocess.Popen([str(root/'.venv-pi05/bin/python'),'-u',str(root/'scripts/pi05_official50/evaluation/campaign.py'),*sys.argv[2:]],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('Campaign PID:',process.pid)
print('Status:',out/'status.json')
print('Log:',out/'campaign.log')
PY
