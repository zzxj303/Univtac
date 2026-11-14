import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from orchestrator.common import ROOT,atomic,read
from adapters.repairs import verify_repair,_replay
from orchestrator.context import build

class RepairExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'runs/repairs');self.p=Path(self.tmp.name)
        self.old=read(ROOT/'runs/univtac_hole_dp_rgbt/round-1/input/context.json')
    def tearDown(self):self.tmp.cleanup()
    def session(self):
        atomic(self.p/'session.json',{'group':self.old['group'],'demos':{'27':{'source_seed':27,'reference':'reference'}},'trials':[]})
    def test_acceptance_requires_independent_success(self):
        self.session();atomic(self.p/'27/original/result.json',{'success':False})
        atomic(self.p/'27/candidate-1/result.json',{'success':True,'tracking_rmse':.01})
        with patch('adapters.repairs._replay',return_value={'success':False,'error':'startup'}):
            self.assertFalse(verify_repair(self.p,27,'candidate-1')['accepted'])
    def test_successful_original_requires_improvement_both_times(self):
        self.session();atomic(self.p/'27/original/result.json',{'success':True,'tracking_rmse':.01})
        atomic(self.p/'27/candidate-1/result.json',{'success':True,'tracking_rmse':.009})
        with patch('adapters.repairs._replay',return_value={'success':True,'tracking_rmse':.02}):
            self.assertFalse(verify_repair(self.p,27,'candidate-1')['accepted'])
    def test_verified_success_accepts_with_provenance(self):
        self.session();atomic(self.p/'27/original/result.json',{'success':True,'tracking_rmse':.02})
        atomic(self.p/'27/candidate-1/result.json',{'success':True,'tracking_rmse':.009})
        with patch('adapters.repairs._replay',return_value={'success':True,'tracking_rmse':.008,'recording':'recording'}):
            a=verify_repair(self.p,27,'candidate-1');self.assertTrue(a['accepted']);self.assertEqual(a['recording'],'recording')
    def test_interrupted_trial_not_replayed(self):
        self.session();m=read(self.p/'session.json');m['trials']=[{'demo_id':27,'kind':'original','folder':str(self.p/'27/original')}];atomic(self.p/'session.json',m)
        with patch('adapters.repairs.run') as run:
            with self.assertRaisesRegex(RuntimeError,'Interrupted'):_replay(self.p,27,'original','commands')
            run.assert_not_called()
    def test_dynamic_sources_exclude_cli_and_process_homes(self):
        self.session()
        atomic(self.p/'harness/attempt-1/codex_home/auth.json',{'not':'real credentials'})
        atomic(self.p/'27/original/process/result.json',{'private':True})
        atomic(self.p/'27/original/result.json',{'success':True})
        from adapters.repairs import repair_sources
        paths=repair_sources(self.p)
        self.assertIn(str(self.p/'27/original/result.json'),paths)
        self.assertFalse(any('codex_home' in p or '/process/' in p for p in paths))

    def test_broker_rejects_arbitrary_operations(self):
        self.session()
        from adapters.repair_broker import start,request
        from orchestrator.common import terminate
        broker,address=start(self.p)
        try:
            with self.assertRaisesRegex(RuntimeError,'Unrecognized operation'):
                request(self.p,'shell',{'command':'not executed'})
            self.assertEqual(read(self.p/'session.json')['trials'],[])
        finally:
            terminate(broker,grace=5)
            if address.exists():address.unlink()

    def test_synthetic_dataset_excluded_from_formal_group(self):
        from adapters.datasets import validate_version
        with tempfile.TemporaryDirectory(dir=ROOT/'runs/datasets') as tmp:
            m=Path(tmp)/'manifest.json'
            atomic(m,{'group':self.old['group']['id'],'task':self.old['group']['task'],'sealed':True,'development_only':True,'train_ids':[]})
            with self.assertRaisesRegex(ValueError,'Synthetic acceptance'):
                validate_version(m,self.old['group'])

    def test_stage_two_uses_second_slot(self):
        from orchestrator import experiment
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime/tmp') as tmp:
            root=Path(tmp);g=self.old['group'];folder=root/'runs'/g['id']/'round-1'
            atomic(folder/'input/context.json',self.old)
            req={'repair_intent':[{'demo_id':27}]};atomic(folder/'harness/attempt-1/requirement.json',req)
            receipt={'termination':None,'verification':{'passed':True},'requirement_file':str(folder/'harness/attempt-1/requirement.json')}
            with patch.object(experiment,'ROOT',root),patch.object(experiment,'call',return_value=receipt),patch.object(experiment,'spent',return_value={'calls':1,'requests':4,'tokens':100,'complete':True}),patch.object(experiment,'within',return_value=True),patch('orchestrator.repair_stage.execute',return_value=({'decision':'stop'},None)) as execute:
                result=experiment.diagnose(g,1,self.old['rounds'],0,1000)
                self.assertEqual(result,({'decision':'stop'},None));self.assertEqual(execute.call_args.args[4],folder/'harness/attempt-2')
if __name__=='__main__':unittest.main()
