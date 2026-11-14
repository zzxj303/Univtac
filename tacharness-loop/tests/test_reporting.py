import unittest
from unittest.mock import patch
from orchestrator.ablation import run
from orchestrator.common import ROOT, read


class ReportTests(unittest.TestCase):
    def test_offline_collection_never_calls_model_and_counts_missing(self):
        # An integration check over the retained real model receipts.
        if len(list((ROOT / 'runs').glob('univtac*/round-1/raw/attempt-1/receipt.json'))) != 18:
            self.skipTest('Full paired diagnosis receipts not available on this server')
        with patch('orchestrator.ablation.call', side_effect=AssertionError('Offline collection called the model')):
            report = run(2, collect_only=True)
        for summary in report['summary'].values():
            self.assertEqual(summary['calls'], 18)
            self.assertEqual(summary['fact_comparable_calls'] + summary['schema_invalid_or_missing_calls'], 18)
            self.assertLessEqual(summary['mean_per_call_evidence_valid_rate'], summary['evidence_valid_rate'])


if __name__ == '__main__':
    unittest.main()
