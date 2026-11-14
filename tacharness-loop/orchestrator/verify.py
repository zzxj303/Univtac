import json, math
from pathlib import Path
import jsonschema
from orchestrator.common import ROOT, read
from adapters.registry import get_adapter

def pointer(doc, field):
    if field == '':
        return doc
    if not field.startswith('/'):
        raise ValueError('JSON field must be an RFC6901 pointer')
    for part in field[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        doc = doc[int(part)] if isinstance(doc, list) else doc[part]
    return doc

def cited_value(source, field):
    if field.startswith('hdf5:'):
        from harness.raw_data import cited
        return cited(source, field)
    if field.startswith('line:'):
        n = int(field.split(':')[1])
        lines = Path(source).read_text().splitlines()
        if n < 1:
            raise ValueError('Lines are 1-based')
        return lines[n - 1]
    return pointer(read(source), field)

def json_equal(actual, expected):
    # JSON booleans are distinct from numbers, including inside containers.
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(json_equal(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(json_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected

def verify(req, context):
    errors = []
    valid = 0
    fact_errors = []
    v3 = context.get('prompt_version') == 'demo-repair-v3'
    if v3 and context.get('repair_session'):
        from adapters.repairs import repair_sources
        context = {**context,'allowed_files':context['allowed_files']+repair_sources(context['repair_session'])}
    try:
        jsonschema.validate(req, read(ROOT / ('harness/schemas/requirement-v3.schema.json' if v3 else 'harness/schemas/requirement.schema.json')))
    except (jsonschema.ValidationError, TypeError) as e:
        return dict(passed=False, reasons=['schema: ' + str(e)], evidence_valid=0, evidence_total=0, fact_errors=[], schema_valid=False)
    allowed = {str(Path(p).resolve()) for p in context['allowed_files']}
    if not req['evidence']:
        errors.append('Evidence must not be empty, including stop')
    for i, e in enumerate(req['evidence']):
        try:
            if str(Path(e['source']).resolve()) not in allowed:
                raise ValueError('Evidence path outside current group allowlist')
            expected = json.loads(e['value_json'])
            actual = cited_value(e['source'], e['field'])
            if not json_equal(actual, expected):
                raise ValueError(f'Expected {expected!r}, actual {actual!r}')
            valid += 1
        except Exception as ex:
            errors.append(f'evidence[{i}]: {ex}')
    adapter = get_adapter(context['group'])
    gold = adapter.facts(context['group'], context['current']['artifact']['id'], context['current']['feedback'], context['seeds'], context['current']['artifact']['train_dir'])
    for k, value in gold.items():
        if k == 'unavailable_reason':
            if value is None and req['facts'].get(k) is not None:
                fact_errors.append(k)
            elif value is not None and (not req['facts'].get(k)):
                fact_errors.append(k)
        elif req['facts'].get(k) != value:
            fact_errors.append(k)
    if fact_errors:
        errors.append('Incorrect facts: ' + ', '.join(fact_errors))
    action = req['decision']
    params = req['parameters']
    artifacts = {key: a for a in context['artifacts'] for key in (a['id'], a['checkpoint'])}
    if action == 'stop':
        if params is not None or req['init_from'] is not None or (not req['stop_reason']):
            errors.append('stop requires no parameters/init_from and a reason')
    else:
        if req['init_from'] not in artifacts:
            errors.append('Unknown init_from')
        if params is None:
            errors.append('Update requires parameters')
        elif action == 'select_ckpt':
            ids = params['checkpoint_ids']
            if not 1 <= len(ids) <= 3 or len(set(ids)) != len(ids) or (not set(ids) <= set(artifacts)):
                errors.append('Select 1..3 distinct known artifacts')
            elif len({artifacts[i]['id'] for i in ids}) != len(ids):
                errors.append('Checkpoint aliases refer to the same artifact')
            if params['steps'] is not None or params['lr_scale'] is not None or params['train_ids']:
                errors.append('Selection cannot train')
        else:
            steps = params['steps']
            original = context['group']['original_steps']
            if type(steps) != int or steps <= 0 or steps > math.floor(0.1 * original) or (context['training_used'] + steps > math.floor(0.25 * original)):
                errors.append('Training update budget exceeded')
            scale = params['lr_scale']
            if scale is not None and (not math.isfinite(scale) or not 0.1 <= scale <= 10):
                errors.append('lr_scale must be within [0.1,10]')
            if params['checkpoint_ids']:
                errors.append('Training cannot select multiple checkpoints')
            ids = params['train_ids']
            if action == 'finetune_more' and ids:
                errors.append('finetune_more uses unchanged data')
            if action == 'finetune_data':
                errors.extend(adapter.validate_training_data(context['group'], ids))
    if v3:
        errors.extend(verify_inspection(req, context))
    return dict(passed=not errors, reasons=errors, evidence_valid=valid, evidence_total=len(req['evidence']), fact_errors=fact_errors, schema_valid=True)


def verify_inspection(req, context):
    errors=[]
    from adapters.registry import get_adapter
    adapter=get_adapter(context['group'])
    demos=context.get('demonstrations')
    if demos is None: demos=adapter.demonstrations(context['group']) if hasattr(adapter,'demonstrations') else []
    demo_paths={str(Path(r[k]).resolve()) for r in demos for k in ('raw_path','training_path') if r.get(k)}
    if context.get('repair_session'):
        from adapters.repairs import repair_sources
        demo_paths.update(p for p in repair_sources(context['repair_session']) if Path(p).suffix == '.hdf5')
    logs={p for p in context['allowed_files'] if Path(p).name in ('metrics.jsonl','train.log')}
    trajectories={p for p in context['allowed_files'] if Path(p).name=='trajectory.jsonl'}
    classes={'demonstrations':demo_paths,'training':logs,'feedback':trajectories}
    for kind,paths in classes.items():
        entry=req['inspection'][kind]; indices=entry['evidence_indices']
        if any(type(i)!=int or i<0 or i>=len(req['evidence']) for i in indices):
            errors.append('Invalid inspection evidence index: '+kind);continue
        actual=[req['evidence'][i] for i in indices if str(Path(req['evidence'][i]['source']).resolve()) in paths]
        if paths and not actual:errors.append('Missing '+kind+' inspection evidence')
        if not paths and not entry['unavailable_reason']:errors.append('Missing unavailable reason: '+kind)
    if context.get('_tool_events'):
        events=[__import__('json').loads(x) for x in Path(context['_tool_events']).read_text().splitlines()]
        executed={e['number'] for e in events if e['event']=='executed'}
        reads=[e['args'] for e in events if e['event']=='requested' and e['number'] in executed and e['name']=='raw.read']
        paths={str(Path(a['path']).resolve()) for a in reads}
        for kind in classes:
            indices=req['inspection'][kind]['evidence_indices']
            if indices and not any(0<=i<len(req['evidence']) and str(Path(req['evidence'][i]['source']).resolve()) in paths for i in indices):
                errors.append('Inspection has no executed raw read: '+kind)
        if demo_paths and not any(a.get('frames') and str(Path(a['path']).resolve()) in demo_paths for a in reads):
            errors.append('Missing demonstration image inspection')
    intents=req['repair_intent']
    if intents:
        if context.get('stage','diagnose')!='diagnose' or req['decision']!='finetune_data':errors.append('Repair intent only allowed during initial finetune_data diagnosis')
        if not 1<=len(intents)<=3 or len({i['demo_id'] for i in intents})!=len(intents):errors.append('Invalid repair demonstration count')
        by={r['demo_id']:r for r in demos}
        import h5py
        for i in intents:
            r=by.get(i['demo_id'])
            if not r or not r['eligible_for_repair']:errors.append('Ineligible repair demonstration');continue
            if i['demo_id'] not in (req['parameters'] or {}).get('train_ids',[]):errors.append('Repaired demo must be in proposed train_ids')
            with h5py.File(r['training_path'] or r['raw_path'],'r') as f:n=len(f['action']) if 'action' in f else len(f['embodiment/joint'])-1
            if not 0<=i['start']<i['stop']<=n:errors.append('Invalid repair window')
    params=req['parameters']
    if params and params.get('dataset_version'):
        try:
            from adapters.datasets import validate_version
            validate_version(params['dataset_version'],context['group'],params['train_ids'] or None)
        except Exception as e:errors.append('Dataset version: '+str(e))
    return errors
