import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from adapters.univtac import UniVTAC
from orchestrator.common import ROOT, atomic, read
from orchestrator.experiment import group_loop


class RecoveryTests(unittest.TestCase):
    def test_training_directory_cannot_change_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime/tmp') as tmp:
            out = Path(tmp)
            atomic(out / 'request.json', {'steps': 50})
            atomic(out / 'artifact.json', {'id': 'already-trained'})
            with self.assertRaisesRegex(RuntimeError, 'Training identity changed'):
                UniVTAC().train({'policy': 'act'}, {'id': 'source'}, {'steps': 100}, out)

    def test_round_commit_recovers_without_reexecution(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime/tmp') as tmp:
            root = Path(tmp)
            art = {'id': 'g:baseline'}
            g = {'id': 'g', 'baseline': art}
            atomic(root / 'configs/token_budget.json', {'tokens_per_group': 100000})
            atomic(root / 'runs/g/rounds.json', [{'round': 0, 'artifact': art, 'feedback': 'b', 'successes': 0}])
            committed = {'round': 1, 'artifact': {'id': 'g:new'}, 'feedback': 'f', 'successes': 2, 'training_updates': 50, 'eligible': True}
            atomic(root / 'runs/g/round-1/round.json', committed)
            called_rounds = []
            def diagnose(group, number, rounds, used, budget):
                called_rounds.append(number)
                self.assertEqual(used, 50)
                return {'decision': 'stop'}, None
            with patch('orchestrator.experiment.ROOT', root), patch('orchestrator.experiment.diagnose', diagnose), patch('orchestrator.experiment.audit'), patch('orchestrator.experiment.test_once', return_value={'reused': False}), patch.object(UniVTAC, 'train', side_effect=AssertionError('Repeated committed training')):
                gate = group_loop(g)
            self.assertEqual(called_rounds, [2])
            self.assertEqual(gate['selected_round'], 1)
            self.assertEqual(len(read(root / 'runs/g/rounds.json')), 2)

class ProtectedDirectoryStopTests(unittest.TestCase):
    def test_protected_change_sets_persistent_halt(self):
        import os
        from orchestrator.common import audit, check_halt
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime/tmp') as tmp:
            root = Path(tmp)
            marker = root / '.start_marker'
            marker.touch()
            os.utime(marker, (1, 1))
            protected = root / 'mock-protected'
            protected.mkdir()
            (protected / 'changed.txt').write_text('only a local test fixture')
            with patch('orchestrator.common.ROOT', root), patch('orchestrator.common.PROTECTED', [protected]):
                with self.assertRaisesRegex(RuntimeError, 'Original directory changed'):
                    audit()
                self.assertTrue((root / 'runs/STOP.json').exists())
                with self.assertRaisesRegex(RuntimeError, 'Run halted'):
                    check_halt()


if __name__ == '__main__':
    unittest.main()
