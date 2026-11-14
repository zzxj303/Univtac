from __future__ import annotations
import json, os, selectors, shutil, subprocess, time
from pathlib import Path
from orchestrator.common import ROOT, PYTHON, read, atomic, environment, confine_writes, terminate, check_halt
from orchestrator.verify import verify
MODEL = 'gpt-6-astra'
EFFORT = 'medium'
CODEX = os.environ.get('TH_CODEX_BIN', '/usr/local/bin/codex')

def usage(home):
    ids = set()
    usage_records = {}
    last = None
    model = None
    effort = None
    paths = list((Path(home) / 'sessions').glob('**/rollout-*.jsonl'))
    broken = False
    for path in paths:
        with path.open(errors='replace') as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    broken = True
                    continue
                p = r.get('payload') or {}
                kind = r.get('type')
                if kind == 'turn_context':
                    model = p.get('model', model)
                    effort = p.get('effort', p.get('reasoning_effort', effort))
                if kind == 'token_usage_record':
                    rid = p.get('response_id')
                    if rid:
                        ids.add(rid)
                        usage_records[rid] = p
                if kind == 'compacted' and p.get('compaction_response_id'):
                    ids.add(p['compaction_response_id'])
                if kind == 'event_msg' and p.get('type') == 'token_count' and p.get('info'):
                    last = p['info'].get('total_token_usage')
    return {'model_requests': len(ids) if ids else None, 'request_ids': sorted(ids), 'input_tokens': last.get('input_tokens', 0) if last else None, 'cached_input_tokens': last.get('cached_input_tokens', 0) if last else None, 'output_tokens': last.get('output_tokens', 0) if last else None, 'total_tokens': last.get('input_tokens', 0) + last.get('output_tokens', 0) if last else None, 'actual_model': model, 'actual_effort': effort, 'metering_complete': bool(ids and last and (not broken)), 'rollouts': [str(p) for p in paths], 'record_fields': sorted({k for v in usage_records.values() for k in v})}

def _call(context_path, folder, condition='harness', reason='initial', correction=None, limits=None):
    check_halt()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'receipt.json').exists():
        return read(folder / 'receipt.json')
    if (folder / 'started.json').exists():
        raise RuntimeError('Interrupted diagnosis must be reconciled; not silently repeated')
    c = read(context_path)
    repair = c.get('stage') == 'repair'
    v3 = c.get('prompt_version') == 'demo-repair-v3'
    home = folder / 'codex_home'
    home.mkdir()
    auth = Path(os.environ.get('TH_CODEX_AUTH', '/root/.codex/auth.json'))
    if not auth.is_file():
        raise RuntimeError('Codex credentials unavailable; no installation or login attempted')
    shutil.copyfile(auth, home / 'auth.json')
    os.chmod(home / 'auth.json', 384)
    config = 'model = "' + MODEL + '"\nmodel_reasoning_effort = "medium"\nweb_search = "disabled"\n'
    config += '[mcp_servers.tacharness]\ncommand = ' + json.dumps(str(PYTHON)) + '\nargs = ' + json.dumps(['-m', 'harness.mcp_server', '--context', str(context_path), '--condition', condition, '--receipt', str(folder / 'tools.jsonl')]) + '\ncwd = ' + json.dumps(str(ROOT)) + '\nstartup_timeout_sec = 60\ntool_timeout_sec = ' + str(1900 if repair else 65) + '\n'
    config += '[mcp_servers.tacharness.env]\n'
    env = environment({'CODEX_HOME': home})
    for key in ['PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'TMPDIR', 'XDG_CACHE_HOME', 'HF_HUB_OFFLINE']:
        config += key + ' = ' + json.dumps(env[key]) + '\n'
    (home / 'config.toml').write_text(config)
    from orchestrator.context import load_prompt, prompt_version
    c = read(context_path)
    text = load_prompt(context_path)
    if correction:
        text += '\nCorrection: previous requirement and verification rejection follow. Correct once.\n' + json.dumps(correction, ensure_ascii=False)
    (folder / 'prompt.txt').write_text(text)
    output = folder / 'requirement.json'
    cmd = [CODEX, 'exec', '--skip-git-repo-check', '-C', str(ROOT / 'harness'), '-s', 'read-only', '-m', MODEL, '-c', 'model_reasoning_effort="medium"', '--output-schema', str(ROOT / ('harness/schemas/requirement-v3.schema.json' if v3 else 'harness/schemas/requirement.schema.json')), '-o', str(output), '--json']
    for feature in ['shell_tool', 'unified_exec', 'apps', 'plugins', 'multi_agent', 'hooks', 'view_image', 'browser_use', 'computer_use', 'image_generation', 'goals', 'sleep_tool', 'skill_search']:
        cmd += ['--disable', feature]
    cmd += ['--enable', 'skip_host_skill_discovery', '-']
    version = subprocess.check_output([CODEX, '--version'], env=env, text=True).strip()
    started = time.time()
    termination = None
    sel = selectors.DefaultSelector()
    events = []
    with (folder / 'stderr.log').open('w') as err, (folder / 'events.jsonl').open('w') as eventlog:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, start_new_session=True, preexec_fn=confine_writes)
        atomic(folder / 'started.json', {'pid': p.pid, 'started_at': started, 'command': cmd}, exclusive=True)
        p.stdin.write(text)
        p.stdin.close()
        sel.register(p.stdout, selectors.EVENT_READ)
        while True:
            if (ROOT / 'runs/STOP.json').exists():
                termination = 'protected_directory_changed'
                terminate(p)
                break
            if time.time() - started > (18000 if repair else 1200):
                termination = 'timeout'
                terminate(p)
                break
            tools = folder / 'tools.jsonl'
            if tools.exists() and 'continued_after_limit' in tools.read_text():
                termination = 'tool_limit'
                terminate(p)
                break
            u = usage(home)
            if limits and ((u['model_requests'] or 0) > limits.get('requests', 150) or (limits.get('tokens') is not None and (u['total_tokens'] or 0) > limits['tokens'])):
                termination = 'budget_limit'
                terminate(p)
                break
            for key, _ in sel.select(0.25):
                line = key.fileobj.readline()
                if not line:
                    sel.unregister(key.fileobj)
                    continue
                eventlog.write(line)
                eventlog.flush()
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                events.append(e)
                if (e.get('item') or {}).get('type') == 'command_execution':
                    termination = 'isolation_violation'
                    terminate(p)
                    break
            if termination or (p.poll() is not None and (not sel.get_map())):
                break
        if p.poll() is None:
            p.wait()
    u = usage(home)
    verification = None
    if output.exists():
        try:
            verification = verify(read(output), {**c, **({'_tool_events':str(folder / 'tools.jsonl')} if v3 and (folder / 'tools.jsonl').exists() else {})})
        except Exception as e:
            verification = {'passed': False, 'reasons': [repr(e)]}
    if termination is None and p.returncode != 0:
        termination = 'runtime_error'
    if termination is None and (not output.exists()):
        termination = 'missing_output'
    if termination is None and (not u['metering_complete']):
        termination = 'metering_incomplete'
    if termination is None and (u['actual_model'] != MODEL or u['actual_effort'] != EFFORT):
        termination = 'model_mismatch'
    tool_rows = [json.loads(line) for line in (folder / 'tools.jsonl').read_text().splitlines()] if (folder / 'tools.jsonl').exists() else []
    tool_counts = {'requested': sum((t.get('event') == 'requested' for t in tool_rows)), 'executed': sum((t.get('event') == 'executed' for t in tool_rows)), 'rejected_at_limit': sum((t.get('event') == 'requested' and t.get('number', 0) > 40 for t in tool_rows))}
    receipt = {'attempt': int(folder.name.split('-')[-1]) if folder.name.split('-')[-1].isdigit() else folder.name, 'timed_out': termination == 'timeout', 'limit_triggered': termination in ('budget_limit', 'tool_limit'), 'tool_counts': tool_counts, 'group': c['group']['id'], 'round': c['round'], 'reason': reason, 'condition': condition, 'prompt_version': prompt_version(c), 'started_at': started, 'finished_at': time.time(), 'elapsed_s': time.time() - started, 'exit_code': p.returncode, 'termination': termination, 'verification': verification, 'model': MODEL, 'reasoning_effort': EFFORT, 'cli_version': version, 'prompt_file': str(folder / 'prompt.txt'), 'requirement_file': str(output), 'tool_events': str(folder / 'tools.jsonl'), **u}
    atomic(folder / 'receipt.json', receipt, exclusive=True)
    return receipt


def call(context_path, folder, condition='harness', reason='initial', correction=None, limits=None):
    c=read(context_path)
    if c.get('stage') != 'repair' or (Path(folder)/'receipt.json').exists():
        return _call(context_path,folder,condition,reason,correction,limits)
    from adapters.repair_broker import start
    broker,address=start(c['repair_session'])
    try:
        return _call(context_path,folder,condition,reason,correction,limits)
    finally:
        terminate(broker,grace=10)
        if address.exists():address.unlink()
