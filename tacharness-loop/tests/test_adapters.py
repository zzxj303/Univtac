import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from orchestrator.common import ROOT, atomic
from adapters.registry import get_adapter
from orchestrator.context import build


class AdapterContractTests(unittest.TestCase):
    def test_unregistered_backend_is_explicit(self):
        with self.assertRaisesRegex(RuntimeError, 'No local adapter'):
            get_adapter('not_installed')

    def test_context_uses_backend_sources_and_checkpoints(self):
        class Fake:
            def raw_files(self, group, rounds):
                return ['/backend/source.json']
            def checkpoints(self, group):
                return [{'id': 'alternate', 'checkpoint': '/backend/checkpoint'}]
        group = {'benchmark': 'external', 'feedback_seeds': [10001]}
        current = {'artifact': {'id': 'current'}, 'feedback': '/backend/feedback'}
        with patch('orchestrator.context.get_adapter', return_value=Fake()):
            ctx = build(group, [current], 2, 70)
        self.assertEqual(ctx['allowed_files'], ['/backend/source.json'])
        self.assertEqual([a['id'] for a in ctx['artifacts']], ['alternate', 'current'])
        self.assertEqual(ctx['seeds'], [10001])
        self.assertEqual(ctx['training_used'], 70)

    def test_finetune_data_rejects_missing_metadata(self):
        g = {'task': 'task', 'feedback_seeds': [20001], 'test_seeds': [20031]}
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime/tmp') as tmp:
            source = Path(tmp)
            atomic(source / 'data/official50/task/split.json', {'validation_ids': [1]})
            atomic(source / 'data/isaac51/task/metadata.json', {'1': {'seed': 1, 'source_seed': 1}})
            with patch('adapters.univtac.SOURCE', source):
                errors = get_adapter().validate_training_data(g, [50])
        self.assertIn('Demonstration seed metadata missing', errors)


if __name__ == '__main__':
    unittest.main()
