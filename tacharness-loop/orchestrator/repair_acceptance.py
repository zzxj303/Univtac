"""Explicitly synthetic local-corruption acceptance, never a formal improvement."""
import json,time
from pathlib import Path
from orchestrator.common import ROOT,read,atomic,lock
from orchestrator.context import build,save
from orchestrator.codex import call
from adapters.datasets import catalog,physical_copy
from adapters.repairs import patch_commands,_replay,repair_sources

FOLDER=ROOT/'runs/repairs/acceptance-v3/synthetic-local-offset'

def main():
    old=read(ROOT/'runs/univtac_hole_dp_rgbt/round-1/input/context.json');g=old['group'];i=27
    FOLDER.mkdir(parents=True,exist_ok=True)
    with lock(FOLDER/'.acceptance.lock'):
        if not (FOLDER/'session.json').exists():
            d=FOLDER/str(i);d.mkdir()
            row=next(r for r in catalog(g) if r['demo_id']==i)
            provenance=physical_copy(row['training_path'],d/'reference.hdf5')
            patch={'start':60,'stop':100,'mode':'offset','values':[.03,0,0,0,0,0,0,0]}
            patch_commands(d/'reference.hdf5',d/'source_commands.hdf5',patch)
            intent={'demo_id':i,'start':60,'stop':100,'issue':'SYNTHETIC TECHNICAL ACCEPTANCE ONLY: owned command copy has a deliberately injected tapered +0.03 rad joint-0 offset over [60,100). Compare reference and source_commands, remove this corruption locally and verify in physics. This is not evidence of an original training-data defect.','goal':'tracking_rmse'}
            atomic(FOLDER/'fixture.json',{'scope':'synthetic technical acceptance only','injected_patch':patch,'reference':str(d/'reference.hdf5'),'corrupted_copy':str(d/'source_commands.hdf5'),'copy':provenance},exclusive=True)
            atomic(FOLDER/'session.json',{'group':g,'intents':[intent],'parent':None,'demos':{'27':{**row,'reference':str(d/'reference.hdf5'),'base_commands':str(d/'source_commands.hdf5'),'intent':intent}},'trials':[],'created_at':time.time()},exclusive=True)
        with lock(FOLDER/'.lock'):
            _replay(FOLDER,27,'original',FOLDER/'27/source_commands.hdf5')
        inputs=FOLDER/'diagnosis-input'
        if not (inputs/'context.json').exists():
            c=build(g,old['rounds'],1);c.update(stage='repair',repair_session=str(FOLDER))
            c['allowed_files']+= repair_sources(FOLDER)
            save(inputs,c)
        latest=FOLDER/'harness/attempt-2/receipt.json'
        r=read(latest) if latest.exists() else call(inputs/'context.json',FOLDER/'harness/attempt-1',reason='synthetic_repair_acceptance',limits={'requests':150,'tokens':1800000})
        print(json.dumps({k:r[k] for k in ['termination','verification','tool_counts','total_tokens']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
