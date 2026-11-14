"""Second CLI slot for demonstration repair; original calibration stays GPU-free."""
from pathlib import Path
from orchestrator.common import ROOT,read,atomic
from orchestrator.context import save
from adapters.repairs import prepare,accepted_replacements,repair_sources
from adapters.datasets import create_version

def execute(g,k,input_dir,requirement,call_folder,limits):
    from orchestrator.codex import call
    from adapters.registry import get_adapter
    c=read(Path(input_dir)/'context.json')
    session=ROOT/'runs/repairs'/g['id']/str(Path(input_dir).parent.relative_to(ROOT/'runs')).replace('/','_')
    parent=c['current']['artifact'].get('dataset_version')
    adapter=get_adapter(g)
    repair=adapter.prepare_repairs(g,requirement['repair_intent'],session,parent=parent)
    d=Path(input_dir).parent/'repair-input'
    if not (d/'context.json').exists():
        c={**c,'stage':'repair','repair_session':repair}
        c['allowed_files']=sorted(set(c['allowed_files']+repair_sources(repair)))
        save(d,c)
    receipt=call(d/'context.json',call_folder,'harness','demonstration_repair',limits=limits)
    if receipt['termination'] or not receipt['verification'] or not receipt['verification']['passed']:
        return None,'repair_diagnosis_failed'
    req=read(receipt['requirement_file'])
    if req['decision']=='finetune_data':
        replacements={i:p for i,p in adapter.accepted_repairs(repair).items() if i in req['parameters']['train_ids']}
        if not replacements:return None,'no_accepted_demonstration_repair'
        version_path=d/'dataset-version.json'
        if version_path.exists():version=read(version_path)['version']
        else:
            version=adapter.create_dataset_version(g,req['parameters']['train_ids'],replacements,parent)
            atomic(version_path,{'version':version},exclusive=True)
        req['parameters']['dataset_version']=version
        atomic(d/'resolved-requirement.json',req)
    return req,None
