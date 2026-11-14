import json
from pathlib import Path
from orchestrator.common import ROOT, read, atomic
from adapters.registry import get_adapter
PROMPT_V1 = 'You diagnose a robot-policy benchmark using only the exposed read-only MCP tools. Do not use shell commands or other tools. Never read test data. Inspect raw sources and provide independently checkable facts. Every requirement, including stop, needs failure category and nonempty evidence. Cite raw JSON with RFC6901 pointers or exact text using field line:N. value_json is the JSON serialization of the cited value, not an explanation. Count policy optimizer updates, not epochs. Return only the requirement JSON matching the provided schema. Choose exactly one permitted update or stop. Do not invent root causes; distinguish observed failure from hypotheses. Statistical uncertainty alone does not establish a causal diagnosis. Predictions are hypotheses, not measured improvements. Stop when no justified budget-compatible update remains. Training starts from a registered checkpoint, keeps normalization, validation set, architecture and inference protocol fixed, resets optimizer, and may change only learning-rate scale, update count, and (finetune_data only) training demonstration IDs. save_at_steps is not an agent parameter. You may use raw.list to inspect the available sources. A source unavailable in the tool allowlist cannot be cited. Use the context below to identify the exact current feedback and training run for facts.\n'

LEGACY_VERSION = 'diagnostic-v1'
DEFAULT_VERSION = 'demo-repair-v3'

def prompt_version(context):
    return context.get('prompt_version', LEGACY_VERSION)

def template(version):
    if version == LEGACY_VERSION:
        return PROMPT_V1
    if version in ('repair-v2', DEFAULT_VERSION):
        return (ROOT / 'harness/prompts' / (version + '.txt')).read_text()
    raise ValueError('Unknown prompt version: ' + str(version))

def load_prompt(context_path):
    """Use the frozen input, including for retries, rather than a later template."""
    context_path = Path(context_path)
    context = read(context_path)
    template(prompt_version(context))  # Reject unknown version even with saved text.
    saved = context_path.with_name('prompt.txt')
    return saved.read_text() if saved.exists() else prompt(context)

def build(group, rounds, round_number, training_used=0, *, prompt_version=DEFAULT_VERSION):
    current = rounds[-1]
    adapter = get_adapter(group)
    files = adapter.raw_files(group, rounds)
    if prompt_version == 'demo-repair-v3':
        files += adapter.demonstration_files(group) if hasattr(adapter, 'demonstration_files') else []
    artifacts = {a['id']: a for a in adapter.checkpoints(group)}
    for r in rounds:
        artifacts[r['artifact']['id']] = r['artifact']
    context = dict(version=1, prompt_version=prompt_version, group=group, round=round_number, rounds=rounds, current=current, seeds=group['feedback_seeds'], training_used=training_used, artifacts=list(artifacts.values()), allowed_files=sorted(set(files)))
    if prompt_version == 'demo-repair-v3':
        demos = adapter.demonstrations(group) if hasattr(adapter,'demonstrations') else []
        version = current['artifact'].get('dataset_version')
        if version:
            from adapters.datasets import validate_version
            m = validate_version(version, group)
            context['allowed_files'].append(version)
            for entry in m['episodes'].values(): context['allowed_files'].append(entry['path'])
            for row in demos:
                entry=m['episodes'].get(str(row['demo_id']))
                if entry:
                    row['training_path']=entry['path']; row['dataset_version']=version
        context['demonstrations']=demos
        context['allowed_files']=sorted(set(context['allowed_files']))
    return context

def prompt(context):
    g = context['group']
    public = {'group': g['id'], 'policy': g['policy'], 'task': g['task'], 'modality': g['modality'], 'round': context['round'], 'current_artifact': context['current']['artifact'], 'feedback_directory': context['current']['feedback'], 'seeds': context['seeds'], 'rounds': context['rounds'], 'artifacts': context['artifacts'], 'original_updates': g['original_steps'], 'training_used': context['training_used'], 'round_update_cap': int(g['original_steps'] * 0.1), 'group_update_cap': int(g['original_steps'] * 0.25), 'max_checkpoints_per_round': 3}
    if prompt_version(context) == 'demo-repair-v3':
        public.update(stage=context.get('stage','diagnose'), repair_session=context.get('repair_session'), dataset_version=context['current']['artifact'].get('dataset_version'))
    return template(prompt_version(context)) + json.dumps(public, ensure_ascii=False, sort_keys=True, indent=2)

def save(folder, context):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    atomic(folder / 'context.json', context, exclusive=True)
    (folder / 'prompt.txt').write_text(prompt(context))
