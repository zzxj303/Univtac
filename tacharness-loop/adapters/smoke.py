import argparse, json
from orchestrator.common import ROOT, read, atomic, audit
from adapters.evidence import discover
from adapters.univtac import UniVTAC

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', choices=['act', 'dp', 'pi05'], required=True)
    p.add_argument('--action', choices=['select_ckpt', 'finetune_more', 'finetune_data'], required=True)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--evaluate', action='store_true')
    a = p.parse_args()
    g = next((g for g in discover() if g['target'] and g['policy'] == a.policy))
    adapter = UniVTAC()
    out = ROOT / 'runs/acceptance' / a.policy / a.action
    if a.action == 'select_ckpt':
        art = g['baseline']
    else:
        from orchestrator.common import SOURCE
        split = read(SOURCE / 'data/official50' / g['task'] / 'split.json')
        ids = split['train_ids'][:-1] + [50] if a.action == 'finetune_data' else []
        art = adapter.train(g, g['baseline'], {'steps': a.steps, 'lr_scale': 1.0, 'train_ids': ids}, out, save_at_steps=[max(1, a.steps // 2), a.steps])
    if a.evaluate:
        summary = adapter.evaluate(g, art, [20001, 20002], out / 'feedback')
        print(json.dumps(summary))
    atomic(out / 'smoke.json', {'training_or_selection_passed': True, 'evaluated': a.evaluate, 'artifact': art})
    audit()
if __name__ == '__main__':
    main()
