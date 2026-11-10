"""One paper-budget run, with restartable optimizer state and live metrics."""
from pathlib import Path
import argparse
import fcntl
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from data import make_dataset, EOS
from model import ToyModel


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def learning_rate(step, total_steps, warmup=500):
    if step < warmup:
        return 1e-3 * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup - 1)
    return 1e-3 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1))))


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=512, masked=False, save_predictions=None):
    model.eval()
    metrics, predictions = {}, {}
    names = sorted(key[:-7] for key in dataset.files if key.endswith('_answer'))
    for name in names:
        if masked and not name.startswith(('arithmetic_', 'semantic_')):
            continue
        outputs = []
        for start in range(0, len(dataset[name + '_x']), batch_size):
            x = torch.as_tensor(dataset[name + '_x'][start:start + batch_size], device=device)
            pos = torch.as_tensor(dataset[name + '_pos'][start:start + batch_size], device=device)
            edge = torch.as_tensor(dataset[name + '_edge'][start:start + batch_size], device=device) if masked else None
            logits = model(x, blocked=edge)
            pred = logits[torch.arange(len(x), device=device), pos].argmax(-1)
            outputs.append(pred.cpu().numpy())
        pred = np.concatenate(outputs)
        gold, first = dataset[name + '_answer'], dataset[name + '_first']
        correct = pred == gold
        wrong_first = (pred == first) & ~correct
        eos = (pred == EOS) & ~correct & ~wrong_first
        other = ~(correct | wrong_first | eos)
        counts = dict(correct=int(correct.sum()), first=int(wrong_first.sum()),
                      eos=int(eos.sum()), other=int(other.sum()))
        metrics[name] = {'n': len(pred), 'accuracy': float(correct.mean()),
                         'counts': counts, 'rates': {k: v / len(pred) for k, v in counts.items()}}
        predictions[name] = pred
    for group in ['arithmetic', 'semantic']:
        subsets = [v for k, v in metrics.items() if k.startswith(group + '_')]
        n = sum(v['n'] for v in subsets)
        counts = {k: sum(v['counts'][k] for v in subsets) for k in ['correct', 'first', 'eos', 'other']}
        metrics[group] = {'n': n, 'counts': counts, 'accuracy': counts['correct'] / n,
                          'rates': {k: v / n for k, v in counts.items()}}
    if save_predictions:
        np.savez_compressed(save_predictions, **predictions)
    model.train()
    return metrics


def run(args):
    root = Path(args.root)
    run_dir = root / 'runs' / f'{args.architecture}_seed{args.seed}'
    if args.small:
        run_dir = root / 'smoke' / f'{args.architecture}_seed{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    run_lock = (run_dir / 'run.lock').open('w')
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(args.cpu_threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; refusing to silently fall back to CPU.')
    device = torch.device(args.device)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    data_path = make_dataset(args.seed, root / 'datasets', args.small)
    dataset = np.load(data_path)
    model = ToyModel(args.architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                                  weight_decay=0.1, fused=device.type == 'cuda')
    config = {k: v for k, v in vars(args).items() if k != 'root'} | {'parameters': sum(p.numel() for p in model.parameters()),
                          'data': str(data_path.relative_to(root)), 'data_metadata': json.loads(data_path.with_suffix('.json').read_text()),
                          'torch_version': torch.__version__, 'precision': 'float32 with TF32 allowed',
                          'device_name': torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'}
    config_path = run_dir / 'config.json'
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for key in ['architecture', 'seed', 'epochs', 'batch_size', 'microbatch', 'small']:
            if old[key] != config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        if old['data_metadata']['sha256'] != config['data_metadata']['sha256']:
            raise ValueError('Resume dataset checksum mismatch')
    atomic_json(config_path, config)
    checkpoint = run_dir / 'latest.pt'
    start_epoch, step, elapsed_before = 0, 0, 0.0
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch, step, elapsed_before = ckpt['epoch'], ckpt['step'], ckpt['elapsed_seconds']
    arrays = {k: torch.as_tensor(dataset['train_' + k], device=device) for k in ['x', 'y', 'w']}
    n = len(arrays['x'])
    steps_per_epoch = math.ceil(n / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    start_time = time.monotonic()
    latest_metrics = None
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.monotonic()
        generator = torch.Generator().manual_seed(args.seed * 100000 + epoch)
        order = torch.randperm(n, generator=generator).to(device)
        epoch_loss = 0.0
        model.train()
        for offset in range(0, n, args.batch_size):
            idx = order[offset:offset + args.batch_size]
            denominator = arrays['w'][idx].sum()
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(step, total_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            step_loss = torch.zeros((), device=device)
            for micro_start in range(0, len(idx), args.microbatch):
                mi = idx[micro_start:micro_start + args.microbatch]
                logits = model(arrays['x'][mi])
                loss = F.cross_entropy(logits.float().flatten(0, 1), arrays['y'][mi].flatten(), reduction='none')
                loss = (loss.view_as(arrays['w'][mi]) * arrays['w'][mi]).sum() / denominator
                loss.backward()
                step_loss += loss.detach()
            if not torch.isfinite(step_loss):
                raise FloatingPointError(f'Non-finite loss at step {step}')
            optimizer.step()
            epoch_loss += float(step_loss)
            step += 1
        elapsed = elapsed_before + time.monotonic() - start_time
        row = {'epoch': epoch + 1, 'step': step, 'loss': epoch_loss / steps_per_epoch, 'lr': lr,
               'epoch_seconds': time.monotonic() - epoch_start, 'elapsed_seconds': elapsed}
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs or epoch == 0:
            latest_metrics = evaluate(model, dataset, device)
            row['evaluation'] = latest_metrics
        with (run_dir / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        status = {'state': 'training', 'architecture': args.architecture, 'seed': args.seed,
                  'epoch': epoch + 1, 'target_epochs': args.epochs, 'step': step,
                  'elapsed_seconds': elapsed, 'latest': row, 'last_evaluation': latest_metrics}
        atomic_json(run_dir / 'status.json', status)
        print(json.dumps({k: v for k, v in row.items() if k != 'evaluation'}), flush=True)
        if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1, 'step': step, 'elapsed_seconds': elapsed}, checkpoint.with_suffix('.tmp'))
            checkpoint.with_suffix('.tmp').replace(checkpoint)
    baseline = evaluate(model, dataset, device, save_predictions=run_dir / 'predictions.npz')
    masked = (evaluate(model, dataset, device, masked=True, save_predictions=run_dir / 'masked_predictions.npz')
              if args.architecture == 'transformer' else None)
    result = {'architecture': args.architecture, 'seed': args.seed, 'completed_epochs': args.epochs,
              'parameters': config['parameters'], 'elapsed_seconds': elapsed_before + time.monotonic() - start_time,
              'baseline': baseline, 'layer0_attention_edge_masked': masked, 'small_smoke_test': args.small}
    atomic_json(run_dir / 'result.json', result)
    atomic_json(run_dir / 'status.json', {'state': 'complete', 'result': result})
    print('COMPLETE ' + str(run_dir), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='outputs/toy')
    parser.add_argument('--architecture', choices=['transformer', 'hybrid'], required=True)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--batch-size', type=int, default=8190)
    parser.add_argument('--microbatch', type=int, default=2048)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--checkpoint-every', type=int, default=25)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--small', action='store_true')
    run(parser.parse_args())
