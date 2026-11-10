from pathlib import Path
import json
import statistics


def render(root):
    root = Path(root)
    status_file = root / 'scheduler_status.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else {'state': 'not_started'}
    lines = ['# Toy model: Transformer / hybrid comparison', '',
             'This is a new reconstruction from the paper; original code and data were unavailable.',
             'Results are independent of the paper’s historical five-seed results.', '',
             f'Current state: **{status["state"]}**. Updated: {status.get("updated_at", "not started")}.', '',
             'The `qt1` screen session supervises the queue. A queued or CUDA-blocked run has no accuracy result.', '',
             '| Model | Seed | Status | Epoch | Arithmetic | Standard retrieval | Semantic retrieval | Semantic first-token error |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    completed = {'transformer': [], 'hybrid': []}
    for seed in range(101, 106):
        for arch in ['transformer', 'hybrid']:
            d = root / 'runs' / f'{arch}_seed{seed}'
            rf, sf = d / 'result.json', d / 'status.json'
            if rf.exists():
                r = json.loads(rf.read_text()); m = r['baseline']
                state, epoch = 'complete', str(r['completed_epochs'])
                completed[arch].append(r)
            elif sf.exists():
                s = json.loads(sf.read_text()); m = s.get('last_evaluation')
                state, epoch = s['state'], str(s.get('epoch', '—'))
            else:
                state, epoch, m = 'queued', '0', None
            rates = ([f'{100*m["arithmetic"]["accuracy"]:.2f}%', f'{100*m["standard"]["accuracy"]:.2f}%',
                      f'{100*m["semantic"]["accuracy"]:.2f}%', f'{100*m["semantic"]["rates"]["first"]:.2f}%']
                     if m else ['—'] * 4)
            lines.append(f'| {arch} | {seed} | {state} | {epoch} | ' + ' | '.join(rates) + ' |')
    lines += ['', 'Training rows, when present, are interim measurements at the displayed epoch, not final results.', '',
              '## Completed-run summary', '', '| Model | Completed seeds | Arithmetic mean | Standard mean | Semantic mean |',
              '|---|---:|---:|---:|---:|']
    for arch, results in completed.items():
        values = [f'{100*statistics.mean(r["baseline"][k]["accuracy"] for r in results):.2f}%'
                  for k in ['arithmetic', 'standard', 'semantic']] if results else ['—'] * 3
        lines.append(f'| {arch} | {len(results)}/5 | ' + ' | '.join(values) + ' |')
    lines += ['', '## GPU availability', '', '| Physical GPU | Memory used (MiB) | Utilization | State |',
              '|---:|---:|---:|---|']
    for gpu in status.get('gpus', []):
        lines.append(f'| {gpu["index"]} | {gpu["memory_used"]} | {gpu["utilization"]}% | {gpu.get("state", "detected")} |')
    if status.get('cuda_error'):
        lines += ['', 'Last CUDA probe error:', '', '```text', status['cuda_error'], '```']
    lines += ['', 'Full per-pair outcomes, predictions, and layer-0 attention-edge ablations are saved under each completed run.',
              'The hybrid’s convolution remains active during attention-edge ablation; this is not removal of all local information paths.', '']
    tmp = root / 'results.md.tmp'
    tmp.write_text('\n'.join(lines)); tmp.replace(root / 'results.md')


if __name__ == '__main__':
    render(Path(__file__).parent)
