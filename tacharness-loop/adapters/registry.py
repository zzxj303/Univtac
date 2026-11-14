"""Trusted benchmark adapter registry, never controlled by a requirement."""
from importlib import import_module
from orchestrator.common import ROOT, read


def get_adapter(group_or_name='univtac'):
    name = group_or_name.get('benchmark', 'univtac') if isinstance(group_or_name, dict) else group_or_name
    if name == 'univtac':
        from adapters.univtac import UniVTAC
        return UniVTAC()
    path = ROOT / 'configs/adapters.json'
    entries = read(path) if path.exists() else {}
    entry = entries.get(name)
    if not entry:
        raise RuntimeError(f'No local adapter registered for {name}; implement BenchmarkAdapter on that benchmark server')
    module, cls = entry.split(':', 1)
    adapter = getattr(import_module(module), cls)()
    required = ['discover', 'baseline_summary', 'checkpoints', 'raw_files', 'episodes', 'curve', 'facts', 'validate_training_data', 'train', 'evaluate']
    if any(not callable(getattr(adapter, method, None)) for method in required):
        raise TypeError('Adapter does not implement the complete BenchmarkAdapter contract')
    if not isinstance(getattr(adapter, 'protocol', None), str) or not adapter.protocol:
        raise TypeError('Adapter needs a nonempty evaluation protocol identifier')
    return adapter
