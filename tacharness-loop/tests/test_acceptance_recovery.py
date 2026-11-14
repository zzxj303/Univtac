import tempfile
import unittest
from pathlib import Path
from orchestrator.common import ROOT, atomic
from orchestrator.acceptance_recovery import recovery_folder


class AcceptanceRecoveryTests(unittest.TestCase):
    def test_never_replaces_policy_or_reset_trial(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime/tmp') as tmp:
            folder = Path(tmp)
            failed = folder / 'failed.json'
            atomic(folder / 'artifact.json', {'id': 'artifact'})
            atomic(folder / 'acceptance_resolution.json', {'status': 'passed_after_startup_recovery',
                'artifact_id': 'artifact', 'original_failed_result': str(failed), 'feedback': str(folder / 'recovery')})
            for phase in ['policy', 'reset', 'load_policy']:
                atomic(failed, {'phase': phase, 'actions': 0, 'trajectory': None, 'videos': []})
                with self.assertRaisesRegex(RuntimeError, 'must never replace'):
                    recovery_folder(folder)
            atomic(failed, {'phase': 'startup', 'actions': 0, 'trajectory': None, 'videos': []})
            self.assertEqual(recovery_folder(folder), folder / 'recovery')


if __name__ == '__main__':
    unittest.main()
