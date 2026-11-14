import json
from pathlib import Path

def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}
s = {'type': 'string'}
integer = {'type': 'integer'}
nullable_int = {'type': ['integer', 'null']}
arr = lambda x: {'type': 'array', 'items': x}
schema = obj({'failure': obj({'category': {'type': 'string', 'enum': ['data', 'training', 'selection', 'control', 'eval_noise']}, 'conclusion': s}), 'evidence': arr(obj({'source': s, 'field': s, 'value_json': s})), 'facts': obj({'feedback_artifact_id': s, 'seeds': arr(integer), 'feedback_successes': integer, 'termination_counts': obj({k: integer for k in ['success', 'max_steps', 'early_stop', 'exception']}), 'training_run': s, 'best_validation_step': nullable_int, 'unavailable_reason': {'type': ['string', 'null']}}), 'decision': {'type': 'string', 'enum': ['select_ckpt', 'finetune_more', 'finetune_data', 'stop']}, 'init_from': {'type': ['string', 'null']}, 'parameters': {'anyOf': [{'type': 'null'}, obj({'steps': nullable_int, 'lr_scale': {'type': ['number', 'null']}, 'checkpoint_ids': arr(s), 'train_ids': arr(integer)})]}, 'prediction': {'type': ['integer', 'null']}, 'stop_reason': {'type': ['string', 'null']}})
(Path(__file__).parent / 'schemas/requirement.schema.json').write_text(json.dumps(schema, indent=2))
