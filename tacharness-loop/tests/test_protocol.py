import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from orchestrator.common import ROOT, atomic, read
from orchestrator.codex import usage
from orchestrator.context import build, prompt
from orchestrator.verify import verify
from adapters.evidence import discover, facts
from harness.mcp_server import Server

class ProtocolTests(unittest.TestCase):

    def test_request_ids_not_token_events(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime/tmp') as d:
            p = Path(d) / 'sessions'
            p.mkdir()
            records = [{'type': 'token_usage_record', 'payload': {'response_id': 'a'}}, {'type': 'token_usage_record', 'payload': {'response_id': 'a'}}, {'type': 'token_usage_record', 'payload': {'response_id': 'b'}}, {'type': 'event_msg', 'payload': {'type': 'token_count', 'info': {'total_token_usage': {'input_tokens': 100, 'cached_input_tokens': 80, 'output_tokens': 10}}}}]
            (p / 'rollout-x.jsonl').write_text('\n'.join((json.dumps(r) for r in records)) + '\n')
            u = usage(d)
            self.assertEqual(u['model_requests'], 2)
            self.assertEqual(u['total_tokens'], 110)
            self.assertEqual(u['cached_input_tokens'], 80)
            self.assertTrue(u['metering_complete'])

    def test_missing_requests_never_guessed_from_tools(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime/tmp') as d:
            self.assertFalse(usage(d)['metering_complete'])
            self.assertIsNone(usage(d)['model_requests'])

    def test_condition_input_identical(self):
        g = next((g for g in discover() if g['target']))
        p = ROOT / 'runs' / g['id'] / 'round-1/input/context.json'
        a = Server(p, 'raw', ROOT / 'runs/tests/raw-tools.jsonl')
        b = Server(p, 'harness', ROOT / 'runs/tests/th-tools.jsonl')
        self.assertEqual(a.ctx, b.ctx)
        self.assertEqual(prompt(a.ctx), prompt(b.ctx))
        self.assertEqual(a.tools(), b.tools()[:2])

    def test_test_seed_denied(self):
        g = next((g for g in discover() if g['target']))
        s = Server(ROOT / 'runs' / g['id'] / 'round-1/input/context.json', 'harness', ROOT / 'runs/tests/test-denied.jsonl')
        with self.assertRaises(PermissionError):
            s.dispatch('th.keyframes', {'seed': 20031})
        with self.assertRaises(PermissionError):
            s.dispatch('raw.read', {'path': str(ROOT / 'runs/private_test' / g['id'] / 'baseline/summary.json')})

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime/tmp') as d:
            p = Path(d) / 'result.json'
            p.symlink_to('/root/.codex/auth.json')
            g = next((g for g in discover() if g['target']))
            s = Server(ROOT / 'runs' / g['id'] / 'round-1/input/context.json', 'raw', ROOT / 'runs/tests/symlink.jsonl')
            with self.assertRaises(PermissionError):
                s.dispatch('raw.read', {'path': str(p)})

    def test_raw_cannot_call_th(self):
        g = next((g for g in discover() if g['target']))
        s = Server(ROOT / 'runs' / g['id'] / 'round-1/input/context.json', 'raw', ROOT / 'runs/tests/raw-denied.jsonl')
        with self.assertRaises(PermissionError):
            s.call('th.curve', {})
if __name__ == '__main__':
    unittest.main()
