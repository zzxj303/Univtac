"""Three native short training runs consume the same verified synthetic repair.
The shared physics acceptance is policy-independent and explicitly development-only.
No result from this module belongs to a formal benchmark experiment.
"""
import json
from pathlib import Path
import h5py,numpy as np
from orchestrator.common import ROOT,SOURCE,read,atomic,audit
from adapters.datasets import create_version,validate_version
from adapters.evidence import discover
from adapters.univtac import UniVTAC

def main():
    proof=ROOT/'runs/repairs/acceptance-v3/synthetic-local-offset/27/acceptance.json'
    accepted=read(proof)
    if not accepted.get('accepted'):raise RuntimeError('No successful independent repair verification; training not authorized by this driver')
    if accepted.get('scope')!='synthetic technical acceptance only':raise ValueError('Expected synthetic acceptance provenance')
    results=[]
    for group in [g for g in discover() if g['target']]:
        group={**group,'_technical_acceptance':True}
        folder=ROOT/'runs/demo-v3-validation/training'/group['policy'];folder.mkdir(parents=True,exist_ok=True)
        ids=read(SOURCE/'data/official50'/group['task']/'split.json')['train_ids']
        if (folder/'dataset-version.json').exists():version=read(folder/'dataset-version.json')['version']
        else:
            version=create_version(group,ids,{27:str(proof)})
            atomic(folder/'dataset-version.json',{'version':version},exclusive=True)
        dataset=validate_version(version,group)
        changed=dataset['episodes']['27']['path']
        with h5py.File(accepted['recording'],'r') as raw,h5py.File(changed,'r') as converted:
            np.testing.assert_array_equal(converted['observations/qpos'][:],raw['embodiment/joint'][:-1,:8])
            np.testing.assert_array_equal(converted['action'][:],raw['embodiment/joint'][1:,:8])
            assert len(converted['observations/images/tac_left'])==len(converted['action'])
        artifact=UniVTAC().train(group,group['baseline'],{'steps':20,'lr_scale':1.0,'train_ids':ids,'dataset_version':version},folder)
        receipt=read(folder/'train/training_receipt.json')
        assert receipt['dataset_sources']['27']==changed
        assert receipt['actual_updates']==20 and receipt['parameter_sample_changed'] and receipt['strict_reload']
        row={'policy':group['policy'],'artifact':artifact,'dataset_version':version,'modified_demo_path':changed,
             'training_receipt':str(folder/'train/training_receipt.json'),'actual_updates':20,'strict_reload':True,'parameter_sample_changed':True,
             'scope':'synthetic development acceptance only; one physics repair reused across policy loaders'}
        results.append(row);atomic(folder/'acceptance.json',row);audit();print(json.dumps(row),flush=True)
    atomic(ROOT/'runs/demo-v3-validation/training/report.json',{'passed':True,'results':results,'policy_feedback_evaluation':False})
if __name__=='__main__':main()
