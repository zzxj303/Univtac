import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import h5py,numpy as np
from orchestrator.common import ROOT,read,atomic
from orchestrator.context import build,save
from orchestrator.verify import verify,cited_value
from adapters.datasets import catalog,physical_copy,create_version,validate_version
from adapters.repairs import patch_commands,try_repair,verify_repair
from adapters.univtac import UniVTAC
from harness.mcp_server import Server

class DemoRepairTests(unittest.TestCase):
    def setUp(self):
        self.old=read(ROOT/'runs/univtac_hole_dp_rgbt/round-1/input/context.json')
        self.group=self.old['group'];self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'runtime/tmp');self.p=Path(self.tmp.name)
        self.ctx=build(self.group,self.old['rounds'],1);save(self.p,self.ctx)
        self.server=Server(self.p/'context.json','raw',self.p/'tools.jsonl')
    def tearDown(self):self.tmp.cleanup()
    def test_catalog_and_hdf5_numeric_image(self):
        rows=self.server.dispatch('raw.list',{'kind':'demos'})['demonstrations']
        self.assertTrue(rows);r=next(x for x in catalog(self.group) if x['demo_id']==27)
        value=self.server.dispatch('raw.read',{'path':r['training_path'],'dataset':'action','start':0,'stop':2})
        self.assertEqual(value['value'],cited_value(value['source'],value['field']))
        image=self.server.dispatch('raw.read',{'path':r['training_path'],'dataset':'observations/images/tac_left','frames':[0]})
        self.assertEqual(image['content'][1]['type'],'image')
        with self.assertRaises(ValueError):self.server.dispatch('raw.read',{'path':r['training_path'],'dataset':'action','start':0,'stop':1000})
    def test_initial_simulation_not_exposed(self):
        self.assertFalse(any(t['name'].startswith('sim.') for t in self.server.tools()))
        with self.assertRaises(PermissionError):self.server.dispatch('sim.try_repair',{})
    def test_validation_not_repairable(self):
        self.assertFalse(next(r for r in catalog(self.group) if r['demo_id']==0)['eligible_for_repair'])
    def test_physical_copy_and_patch_preserve_original(self):
        src=self.p/'source.hdf5'
        data=np.zeros((10,8),dtype='float32')
        with h5py.File(src,'w') as f:f['action']=data
        dst=self.p/'candidate.hdf5'
        patch_commands(src,dst,{'start':2,'stop':8,'mode':'offset','values':[.01]*7+[.001]})
        with h5py.File(src,'r') as f:np.testing.assert_array_equal(f['action'][:],data)
        with h5py.File(dst,'r') as f:
            a=f['action'][:];np.testing.assert_array_equal(a[:2],data[:2]);self.assertGreater(float(a[4,0]),0)
        self.assertNotEqual(src.stat().st_ino,dst.stat().st_ino);self.assertEqual(dst.stat().st_nlink,1)
    def test_hdf5_external_link_rejected(self):
        src=self.p/'links.hdf5'
        with h5py.File(src,'w') as f:f['external']=h5py.ExternalLink('/tmp/no','data')
        with self.assertRaises(ValueError):physical_copy(src,self.p/'copy.hdf5')
    def test_summary_only_coverage_rejected(self):
        r=read(ROOT/'runs/prompt-revisions/repair-v2/univtac_hole_dp_rgbt/harness/attempt-1/requirement.json')
        r['repair_intent']=[];r['parameters']['dataset_version']=None
        r['inspection']={k:{'evidence_indices':[],'unavailable_reason':None} for k in ('demonstrations','training','feedback')}
        result=verify(r,self.ctx)
        self.assertFalse(result['passed']);self.assertIn('Missing demonstrations inspection evidence',result['reasons'])
    def test_actual_attempt_trajectory_access(self):
        folder=ROOT/'runs/acceptance/dp/finetune_more/feedback'
        c=build(self.group,self.old['rounds']+[{'artifact':self.old['current']['artifact'],'feedback':str(folder)}],2)
        path=str(folder/'20001/attempt-1/trajectory.jsonl')
        self.assertIn(path,c['allowed_files'])
    def test_sealed_version_keeps_validation_and_original_read_only(self):
        split=read(Path(self.group['baseline_train'])/'split.json')
        v=create_version(self.group,split['train_ids'])
        m=validate_version(v,self.group)
        self.assertEqual(m['validation_ids'],split['validation_ids']);self.assertFalse(any(x['modified'] for x in m['episodes'].values()))
        self.assertFalse(list(Path(v).parent.glob('*.hdf5')))
    def test_post_candidate_budget_and_unaccepted_data(self):
        # Deterministic transaction test; no Isaac or GPU invocation.
        session=ROOT/'runs/repairs/tests'/self.p.name;session.mkdir(parents=True)
        atomic(session/'session.json',{'demos':{'27':{'intent':{'start':0,'stop':10}}},'trials':[{'demo_id':27,'kind':'candidate-1'},{'demo_id':27,'kind':'candidate-2'}]})
        atomic(session/'27/original/result.json',{'success':False})
        with self.assertRaisesRegex(ValueError,'Two repair candidates'):try_repair(session,27,{})
        atomic(session/'27/candidate-1/result.json',{'success':False})
        with self.assertRaisesRegex(ValueError,'must succeed'):verify_repair(session,27,'candidate-1')
    def test_simulation_failure_is_not_demo_evidence(self):
        session=ROOT/'runs/repairs/tests'/(self.p.name+'-error');session.mkdir(parents=True)
        atomic(session/'session.json',{'demos':{'27':{}},'trials':[]})
        atomic(session/'27/original/result.json',{'success':False,'error':'startup failed'})
        with self.assertRaisesRegex(ValueError,'infrastructure'):try_repair(session,27,{})
if __name__=='__main__':unittest.main()
