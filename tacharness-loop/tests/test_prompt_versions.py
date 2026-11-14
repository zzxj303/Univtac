import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from orchestrator.common import ROOT, read, atomic
from orchestrator.context import DEFAULT_VERSION, LEGACY_VERSION, load_prompt, prompt, prompt_version, save
from harness.mcp_server import Server

class PromptVersionTests(unittest.TestCase):
    def legacy(self):
        return read(ROOT / 'runs/univtac_hole_dp_rgbt/round-1/input/context.json')

    def test_all_existing_initial_inputs_stay_byte_identical(self):
        paths = list((ROOT / 'runs').glob('univtac_*/round-1/input/context.json'))
        self.assertEqual(len(paths), 18)
        for path in paths:
            self.assertEqual(prompt_version(read(path)), LEGACY_VERSION)
            self.assertEqual(prompt(read(path)).encode(), path.with_name('prompt.txt').read_bytes())

    def test_saved_input_survives_template_change_for_retries(self):
        ctx = self.legacy()
        ctx['prompt_version'] = DEFAULT_VERSION
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime/tmp') as d:
            save(d, ctx)
            original = (Path(d) / 'prompt.txt').read_text()
            with patch('orchestrator.context.template', return_value='a later edit'):
                self.assertEqual(load_prompt(Path(d) / 'context.json'), original)

    def test_new_conditions_have_identical_inputs_and_only_tool_difference(self):
        ctx = self.legacy()
        ctx['prompt_version'] = DEFAULT_VERSION
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime/tmp') as d:
            save(d, ctx)
            p = Path(d)
            a = Server(p / 'context.json', 'raw', p / 'raw.jsonl')
            b = Server(p / 'context.json', 'harness', p / 'th.jsonl')
            self.assertEqual(a.ctx, b.ctx)
            self.assertEqual(prompt(a.ctx), prompt(b.ctx))
            self.assertEqual(a.tools(), b.tools()[:2])
            self.assertNotEqual(prompt(a.ctx), prompt(self.legacy()))

    def test_unknown_version_rejected(self):
        ctx = self.legacy()
        ctx['prompt_version'] = 'nonexistent'
        with self.assertRaisesRegex(ValueError, 'Unknown prompt version'):
            prompt(ctx)

    def test_formal_loop_rejects_mismatched_calibration_before_diagnosis(self):
        from orchestrator.experiment import group_loop
        g = self.legacy()['group']
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime/tmp') as d:
            root = Path(d)
            atomic(root / 'configs/token_budget.json', {'tokens_per_group': 1000})
            ctx = self.legacy()
            ctx['prompt_version'] = DEFAULT_VERSION
            atomic(root / 'runs' / g['id'] / 'round-1/input/context.json', ctx)
            with patch('orchestrator.experiment.ROOT', root), patch('orchestrator.experiment.diagnose') as diagnosis:
                with self.assertRaisesRegex(RuntimeError, 'Prompt version differs'):
                    group_loop(g)
                diagnosis.assert_not_called()

if __name__ == '__main__':
    unittest.main()
