import json, unittest
from pathlib import Path
from orchestrator.common import ROOT, read, writable
from orchestrator.verify import verify
from adapters.evidence import discover, facts, episodes, compare
from harness.mcp_server import Server, Exhausted

class CoreTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.group = next((g for g in discover() if g['id'] == 'univtac_hole_act_rgbt'))
        cls.path = ROOT / 'runs' / cls.group['id'] / 'round-1/input/context.json'
        cls.ctx = read(cls.path)

    def valid(self):
        g = self.group
        return {'failure': {'category': 'eval_noise', 'conclusion': 'Insufficient evidence to justify an update'}, 'evidence': [{'source': str(Path(g['baseline_feedback']) / 'summary.json'), 'field': '/counts/success', 'value_json': '10'}], 'facts': facts(g), 'decision': 'stop', 'init_from': None, 'parameters': None, 'prediction': None, 'stop_reason': 'No justified update'}

    def test_baselines(self):
        self.assertEqual(sum((facts(g)['feedback_successes'] for g in discover())), 50)

    def test_valid_stop(self):
        self.assertTrue(verify(self.valid(), self.ctx)['passed'])

    def test_stop_without_evidence(self):
        r = self.valid()
        r['evidence'] = []
        self.assertFalse(verify(r, self.ctx)['passed'])

    def test_independent_fact_errors(self):
        r = self.valid()
        r['facts']['feedback_successes'] = 9
        v = verify(r, self.ctx)
        self.assertEqual(v['evidence_valid'], 1)
        self.assertEqual(v['fact_errors'], ['feedback_successes'])

    def test_independent_citation_errors(self):
        r = self.valid()
        r['evidence'][0]['value_json'] = '9'
        v = verify(r, self.ctx)
        self.assertEqual(v['fact_errors'], [])
        self.assertEqual(v['evidence_valid'], 0)

    def test_agent_save_at_steps_rejected(self):
        r = self.valid()
        r['save_at_steps'] = [10]
        self.assertFalse(verify(r, self.ctx)['schema_valid'])

    def test_raw_tools(self):
        s = Server(self.path, 'raw', ROOT / 'runs/tests/raw.jsonl')
        self.assertEqual([t['name'] for t in s.tools()], ['raw.list', 'raw.read'])
        with self.assertRaises(PermissionError):
            s.dispatch('raw.read', {'path': '/root/.codex/auth.json'})

    def test_tool_limit(self):
        s = Server(self.path, 'harness', ROOT / 'runs/tests/limit.jsonl')
        s.count = 40
        self.assertTrue(s.call('raw.list', {})['isError'])
        with self.assertRaises(Exhausted):
            s.call('raw.list', {})

    def test_json_boolean_is_not_a_number(self):
        from orchestrator.verify import json_equal
        self.assertFalse(json_equal({'counts': [True]}, {'counts': [1]}))
        self.assertTrue(json_equal({'counts': [1.0]}, {'counts': [1]}))

    def test_output_boundary(self):
        with self.assertRaises(PermissionError):
            writable('/root/autodl-tmp/gujie/UniVTAC/not_written')

    def test_pairing(self):
        a = episodes(self.group['baseline_feedback'], self.group['feedback_seeds'])
        self.assertEqual(compare(a, a)['flips'], [])
        with self.assertRaises(ValueError):
            compare(a, a[:-1])
if __name__ == '__main__':
    unittest.main()
