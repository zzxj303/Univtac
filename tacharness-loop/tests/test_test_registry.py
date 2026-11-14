import tempfile, time, unittest
from pathlib import Path
from unittest.mock import patch
from orchestrator.common import ROOT, atomic, read
from orchestrator.experiment import test_once

class TestRegistry(unittest.TestCase):

    def test_baseline_reused_after_gate(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime/tmp') as temp:
            root = Path(temp)
            base = {'id': 'g:baseline', 'stats': 'stats', 'config': 'config'}
            g = {'id': 'g', 'baseline': base, 'test_seeds': [20031]}
            calls = []

            def evaluate(self, group, artifact, seeds, out):
                calls.append(artifact['id'])
                value = {'successes': 1, 'complete': True}
                atomic(out / 'summary.json', value)
                return value
            with patch('orchestrator.experiment.ROOT', root), patch('orchestrator.experiment.UniVTAC.evaluate', evaluate):
                first = test_once(g, base)
                gate = {'gate_id': 'gate-1', 'sealed_at': time.time()}
                second = test_once(g, base, gate)
            self.assertEqual(calls, ['g:baseline'])
            self.assertTrue(second['reused'])
            self.assertEqual(first['registry']['run_id'], second['registry']['run_id'])

    def test_new_artifact_requires_gate_and_order(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime/tmp') as temp:
            root = Path(temp)
            base = {'id': 'g:baseline', 'stats': 'stats', 'config': 'config'}
            new = {**base, 'id': 'g:round1'}
            g = {'id': 'g', 'baseline': base, 'test_seeds': [20031]}

            def evaluate(self, group, artifact, seeds, out):
                value = {'complete': True, 'successes': 0}
                atomic(out / 'summary.json', value)
                return value
            with patch('orchestrator.experiment.ROOT', root), patch('orchestrator.experiment.UniVTAC.evaluate', evaluate):
                with self.assertRaises(ValueError):
                    test_once(g, new)
                gate = {'gate_id': 'gate-2', 'sealed_at': time.time() - 0.1}
                result = test_once(g, new, gate)
            started = read(Path(result['registry']['directory']) / 'test_started.json')
            self.assertGreater(started['started_at'], gate['sealed_at'])
            self.assertEqual(started['gate_id'], 'gate-2')
if __name__ == '__main__':
    unittest.main()
