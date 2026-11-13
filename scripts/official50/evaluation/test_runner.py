"""Synthetic process tests. No simulator, trained model or formal seed is executed."""
import json
import os
import signal
import fcntl
from pathlib import Path
import sys
import tempfile
import unittest
from common import SEEDS, atomic_json, summarize
from runner import run_seed, process_identity


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='univtac-eval-test-')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def command(self, folder, status='success'):
        script = "import json,pathlib; p=pathlib.Path(%r); (p/'worker_result.json').write_text(json.dumps({'seed':20001,'status':%r,'error':None}))" % (str(folder), status)
        return [sys.executable, '-c', script]

    def test_success_and_no_repeat(self):
        p = self.root/'20001'
        first = run_seed(p, {}, 20001, self.command(p), os.environ.copy())
        self.assertEqual(first['status'], 'success')
        second = run_seed(p, {}, 20001, ['/must/not/execute'], os.environ.copy())
        self.assertEqual(first, second)
        summary = summarize(self.root, 'synthetic')
        self.assertEqual(summary['recorded'], 1)
        self.assertAlmostEqual(summary['success_rate_fixed_denominator'], 1/30)
        self.assertFalse(summary['success_rate_is_final'])

    def test_failure_is_not_retried(self):
        p = self.root/'20001'
        self.assertEqual(run_seed(p, {}, 20001, self.command(p, 'failed'), os.environ.copy())['status'], 'failed')
        self.assertEqual(run_seed(p, {}, 20001, ['/no/retry'], os.environ.copy())['status'], 'failed')

    def test_crash_counts_error(self):
        p = self.root/'20001'
        r = run_seed(p, {}, 20001, [sys.executable, '-c', 'raise RuntimeError("test crash")'], os.environ.copy())
        self.assertEqual(r['status'], 'error')

    def test_timeout_and_child_cleanup(self):
        p = self.root/'20001'
        r = run_seed(p, {}, 20001, [sys.executable, '-c', 'import time; time.sleep(60)'], os.environ.copy(), timeout=0.15)
        self.assertEqual(r['status'], 'timeout')
        started = json.loads((p/'started.json').read_text())
        self.assertIsNone(process_identity(started['pid']))
        self.assertEqual(run_seed(p, {}, 20001, ['/no/retry'], os.environ.copy())['status'], 'timeout')

    def test_interrupted_claim_not_relaunched(self):
        p = self.root/'20001';p.mkdir()
        atomic_json(p/'started.json', {'seed':20001,'started':0})
        r = run_seed(p, {}, 20001, ['/must/not/execute'], os.environ.copy())
        self.assertEqual(r['status'], 'interrupted')

    def test_concurrent_group_lock(self):
        lock_path = self.root/'.lock'
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            command = [sys.executable, '-c',
                "import fcntl; f=open(%r,'a'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)" % str(lock_path)]
            import subprocess
            result = subprocess.run(command, capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_interrupt_records_and_cleans_worker(self):
        p = self.root/'20001'
        def interrupt(signum, frame):
            raise KeyboardInterrupt('Synthetic interruption')
        original = signal.signal(signal.SIGALRM, interrupt)
        signal.setitimer(signal.ITIMER_REAL, 0.3)
        try:
            with self.assertRaises(KeyboardInterrupt):
                run_seed(p, {}, 20001, [sys.executable, '-c', 'import time; time.sleep(60)'], os.environ.copy())
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, original)
        self.assertEqual(json.loads((p/'result.json').read_text())['status'], 'interrupted')
        self.assertIsNone(process_identity(json.loads((p/'started.json').read_text())['pid']))

    def test_fixed_denominator_all_outcomes(self):
        for i, seed in enumerate(SEEDS):
            p=self.root/str(seed);p.mkdir()
            atomic_json(p/'result.json', {'seed':seed,'status':['success','failed','error','timeout','interrupted'][i%5]})
        s=summarize(self.root,'synthetic')
        self.assertTrue(s['complete'])
        self.assertEqual(s['exceptions'],18)
        self.assertEqual(s['success_rate_fixed_denominator'],0.2)
        self.assertEqual(s['pending'],0)


if __name__ == '__main__':
    unittest.main()
