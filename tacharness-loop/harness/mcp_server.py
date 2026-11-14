"""Read-only newline JSON-RPC MCP transport, implemented using the standard library."""
import argparse, base64, json, os, signal, sys, time
from pathlib import Path
from orchestrator.common import ROOT, read, append
from adapters.evidence import compare
from adapters.registry import get_adapter

class Exhausted(Exception):
    pass

class Server:

    def __init__(self, context, condition, receipt):
        self.ctx = read(context)
        self.adapter = get_adapter(self.ctx['group'])
        self.condition = condition
        self.receipt = Path(receipt)
        self.count = 0
        self.v3 = self.ctx.get('prompt_version') == 'demo-repair-v3'
        self.allowed = {str(Path(p).resolve()): Path(p) for p in self.ctx['allowed_files']}

    def tools(self):
        specs = [('raw.list', 'List current-group raw source files.', 'list'), ('raw.read', 'Read an allowed raw file. JSON pointer or 1-based lines.', 'read')]
        if self.condition == 'harness':
            specs += [('th.episodes', 'Normalized feedback episodes with raw source references.', 'episodes'), ('th.curve', 'Validation curve, best update, and checkpoint registry.', 'curve'), ('th.compare', 'Paired outcomes and Wilson intervals; identical seeds required.', 'compare'), ('th.keyframes', 'Recorded RGB keyframes. Reports unavailable tactile imagery honestly.', 'keyframes'), ('th.history', 'Prior rounds and executed actions.', 'history')]
        props = {'list': {'offset': {'type': 'integer'}}, 'read': {'path': {'type': 'string'}, 'field': {'type': 'string'}, 'start_line': {'type': 'integer'}, 'limit': {'type': 'integer'}}, 'episodes': {'outcome': {'type': 'string'}, 'termination': {'type': 'string'}, 'offset': {'type': 'integer'}, 'limit': {'type': 'integer'}}, 'curve': {}, 'compare': {'a_round': {'type': 'integer'}, 'b_round': {'type': 'integer'}}, 'keyframes': {'seed': {'type': 'integer'}}, 'history': {}}
        if self.v3:
            props['list'].update(kind={'type':'string'})
            props['read'].update(dataset={'type':'string'},start={'type':'integer'},stop={'type':'integer'},frames={'type':'array','items':{'type':'integer'}},search={'type':'string'},tail={'type':'integer'})
            if self.ctx.get('stage') == 'repair':
                specs += [('sim.try_repair','Execute a bounded local qpos repair in the simulator; at most two candidates per demo.','try'),('sim.verify_repair','Independently replay a successful candidate; at most once per demo.','verify')]
                props['try']={'demo_id':{'type':'integer'},'patch':{'type':'object','properties':{'start':{'type':'integer'},'stop':{'type':'integer'},'mode':{'type':'string','enum':['offset','replace']},'values':{'type':'array','items':{'anyOf':[{'type':'number'},{'type':'array','items':{'type':'number'}}]}}},'required':['start','stop','mode','values'],'additionalProperties':False}}
                props['verify']={'demo_id':{'type':'integer'},'candidate':{'type':'string','enum':['candidate-1','candidate-2']}}
        return [{'name': n, 'description': d, 'inputSchema': {'type': 'object', 'properties': props[k], 'additionalProperties': False}, 'annotations': {'readOnlyHint': not n.startswith('sim.'), 'destructiveHint': False, 'openWorldHint': False}} for n, d, k in specs]

    def call(self, name, args):
        self.count += 1
        started = time.time()
        append(self.receipt, {'event': 'requested', 'number': self.count, 'name': name, 'args': args, 'at': started})
        if self.count == 41:
            return {'content': [{'type': 'text', 'text': '工具额度已用完，请直接输出最终 requirement。'}], 'isError': True}
        if self.count > 41:
            append(self.receipt, {'event': 'continued_after_limit', 'at': time.time()})
            raise Exhausted()
        names = {t['name'] for t in self.tools()}
        if name not in names:
            raise PermissionError('Tool not exposed in this condition')

        def timeout(*_):
            raise TimeoutError('Tool exceeded 60 seconds')
        old = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(1860 if name.startswith('sim.') else 60)
        try:
            result = self.dispatch(name, args)
            append(self.receipt, {'event': 'executed', 'number': self.count, 'name': name, 'seconds': time.time() - started})
            return result if isinstance(result, dict) and 'content' in result else {'content': [{'type': 'text', 'text': json.dumps(result, ensure_ascii=False)}]}
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    def dispatch(self, name, a):
        c = self.ctx
        current = c['current']
        if self.v3 and c.get('repair_session'):
            from adapters.repairs import repair_sources
            for source in repair_sources(c['repair_session']):
                self.allowed[source] = Path(source)
        if name.startswith('sim.'):
            if not self.v3 or c.get('stage') != 'repair': raise PermissionError('Simulation unavailable during initial diagnosis')
            from adapters.repair_broker import request
            if name == 'sim.try_repair': return request(c['repair_session'],'try',a)
            if name == 'sim.verify_repair': return request(c['repair_session'],'verify',a)
            raise ValueError('Unknown simulation tool')
        if name == 'raw.list':
            if self.v3 and a.get('kind') == 'demos':
                rows = c.get('demonstrations', []); offset=a.get('offset',0)
                if offset<0: raise ValueError('Negative offset')
                return {'demonstrations':rows[offset:offset+20],'total':len(rows),'next_offset':offset+20 if offset+20<len(rows) else None}
            offset = a.get('offset', 0)
            if offset < 0:
                raise ValueError('Negative offset')
            paths = sorted(self.allowed)
            return {'files': paths[offset:offset + 100], 'total': len(paths)}
        if name == 'raw.read':
            p = Path(a['path']).resolve()
            if str(p) not in self.allowed:
                raise PermissionError('Path is not in this group raw allowlist')
            if self.v3 and p.suffix in ('.h5','.hdf5'):
                from harness.raw_data import read_hdf5
                return read_hdf5(p,a)
            if self.v3 and (a.get('search') is not None or a.get('tail') is not None):
                lines=p.read_text(errors='replace').splitlines(); needle=a.get('search'); tail=a.get('tail')
                if needle is not None and (not needle or len(needle)>200): raise ValueError('Search requires 1..200 literal characters')
                if tail is not None and not 1<=tail<=200: raise ValueError('Tail requires 1..200 lines')
                start=a.get('start_line',1)
                matches=[{'line':i+1,'text':line} for i,line in enumerate(lines) if i+1>=start and (needle is None or needle in line)]
                rows=matches[-tail:] if tail else matches[:20]
                if sum(len(r['text']) for r in rows)>24000: raise ValueError('Search result too large; narrow selection')
                return {'source':str(p),'lines':rows,'total_lines':len(lines),'matching_lines':len(matches),'next_line':rows[-1]['line']+1 if rows else None}
            if p.suffix not in ['.json', '.jsonl', '.yml', '.yaml', '.log', '.txt', '.csv']:
                raise ValueError('Binary source is not readable as text')
            if a.get('field') is not None:
                from orchestrator.verify import pointer
                result = pointer(read(p), a['field'])
                text = json.dumps(result, ensure_ascii=False)
                if len(text) > 24000:
                    raise ValueError('Field too large; select a narrower JSON pointer')
                return {'source': str(p), 'field': a['field'], 'value': result}
            start = a.get('start_line', 1)
            limit = a.get('limit', 80)
            if start < 1 or not 1 <= limit <= 200:
                raise ValueError('Invalid line range')
            out = []
            size = 0
            with p.open(errors='replace') as f:
                for n, line in enumerate(f, 1):
                    if n < start:
                        continue
                    if len(out) >= limit or size + len(line) > 24000:
                        break
                    out.append({'line': n, 'text': line.rstrip('\n')})
                    size += len(line)
            return {'source': str(p), 'lines': out, **({'next_line':out[-1]['line']+1 if out else None} if self.v3 else {})}
        if name == 'th.history':
            return {'rounds': c['rounds'], 'training_used': c['training_used']}
        if name == 'th.curve' and self.v3 and c['group'].get('benchmark') == 'univtac':
            from adapters.evidence import training_overview
            return {'curve':self.adapter.curve(c['group'],current['artifact']['train_dir']),'training':training_overview(c['group'],current['artifact']['train_dir']),'checkpoints':c['artifacts']}
        if name == 'th.curve':
            return {'curve': self.adapter.curve(c['group'], current['artifact']['train_dir']), 'checkpoints': c['artifacts']}
        if name == 'th.episodes':
            eps = self.adapter.episodes(current['feedback'], c['seeds'])
            selected = [e for e in eps if (not a.get('outcome') or e['outcome'] == a['outcome']) and (not a.get('termination') or e['termination'] == a['termination'])]
            offset = a.get('offset', 0)
            limit = a.get('limit', 10)
            if offset < 0 or not 1 <= limit <= 30:
                raise ValueError('Invalid paging')
            return {'total': len(eps), 'successes': sum((e['outcome'] == 'success' for e in eps)), 'termination_counts': {k: sum((e['termination'] == k for e in eps)) for k in ['success', 'max_steps', 'early_stop', 'exception']}, 'episodes': selected[offset:offset + limit]}
        if name == 'th.compare':
            by = {r['round']: r for r in c['rounds']}
            aa = by[a['a_round']]
            bb = by[a['b_round']]
            return compare(self.adapter.episodes(aa['feedback'], c['seeds']), self.adapter.episodes(bb['feedback'], c['seeds']))
        if name == 'th.keyframes':
            seed = a['seed']
            if seed not in c['seeds']:
                raise PermissionError('Not a feedback seed')
            ep = next((e for e in self.adapter.episodes(current['feedback'], c['seeds']) if e['seed'] == seed))
            if not ep['refs']['videos']:
                return {'available': False, 'reason': 'No recorded video', 'tactile': 'not recorded'}
            import cv2
            cap = cv2.VideoCapture(ep['refs']['videos'][0])
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            content = []
            try:
                for idx in sorted({0, n // 2, max(0, n - 1)}):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if ok:
                        frame = cv2.resize(frame, (640, 360))
                        ok, data = cv2.imencode('.jpg', frame)
                        if ok:
                            content.append({'type': 'image', 'mimeType': 'image/jpeg', 'data': base64.b64encode(data).decode()})
            finally:
                cap.release()
            return {'content': [{'type': 'text', 'text': json.dumps({'source': ep['refs']['videos'][0], 'frames': sorted({0, n // 2, max(0, n - 1)}), 'tactile': 'No dedicated tactile video is recorded'})}] + content}
        raise ValueError('Unknown tool')

def main():
    def stop(*_): raise SystemExit(143)
    signal.signal(signal.SIGTERM, stop)
    p = argparse.ArgumentParser()
    p.add_argument('--context', required=True)
    p.add_argument('--condition', choices=['raw', 'harness'], required=True)
    p.add_argument('--receipt', required=True)
    a = p.parse_args()
    s = Server(a.context, a.condition, a.receipt)
    for line in sys.stdin:
        req = {}
        try:
            req = json.loads(line)
            method = req.get('method')
            params = req.get('params', {})
            if 'id' not in req:
                continue
            if method == 'initialize':
                result = {'protocolVersion': params.get('protocolVersion', '2024-11-05'), 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'tacharness', 'version': '1.0.0'}}
            elif method == 'tools/list':
                result = {'tools': s.tools()}
            elif method == 'tools/call':
                try:
                    result = s.call(params['name'], params.get('arguments', {}))
                except Exhausted:
                    raise SystemExit(42)
                except Exception as e:
                    result = {'content': [{'type': 'text', 'text': str(e)}], 'isError': True}
            elif method == 'ping':
                result = {}
            else:
                raise ValueError('Unsupported method')
            response = {'jsonrpc': '2.0', 'id': req['id'], 'result': result}
        except Exception as e:
            response = {'jsonrpc': '2.0', 'id': req.get('id'), 'error': {'code': -32602, 'message': str(e)}}
        print(json.dumps(response, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
