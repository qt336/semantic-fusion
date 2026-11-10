"""Validate actual parameter sharing, full-loop gradients, causality and paired data."""
from pathlib import Path
import hashlib
import json
import torch
from model import ToyModel, weighted_loss


def main():
    torch.set_num_threads(2)
    torch.manual_seed(101)
    loop = ToyModel('loop', layers=4, loops=2)
    assert len(loop.blocks) == 4
    assert sum(p.numel() for p in loop.parameters()) == 3331920
    torch.manual_seed(101)
    baseline = ToyModel('transformer', layers=4)
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, loop.state_dict()[name]), name
    explicit = ToyModel('transformer', layers=8)
    mapped = {}
    for name in explicit.state_dict():
        source = name
        if name.startswith('blocks.'):
            parts = name.split('.')
            parts[1] = str(int(parts[1]) % 4)
            source = '.'.join(parts)
        mapped[name] = loop.state_dict()[source].clone()
    explicit.load_state_dict(mapped)
    visits = [0] * 4
    handles = []
    for index, block in enumerate(loop.blocks):
        def visited(module, inputs, output, index=index):
            visits[index] += 1
        handles.append(block.register_forward_hook(visited))
    x = torch.randint(0, 336, (3, 11))
    y = torch.randint(0, 336, (3, 11))
    w = torch.ones_like(x, dtype=torch.float32)
    actual, expected = loop(x), explicit(x)
    assert visits == [2, 2, 2, 2], visits
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(actual, expected)
    weighted_loss(actual, y, w).backward()
    weighted_loss(expected, y, w).backward()
    unrolled_parameters = dict(explicit.named_parameters())
    for name, parameter in loop.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        if name.startswith('blocks.'):
            parts = name.split('.')
            parts[1] = str(int(parts[1]) + 4)
            reference = unrolled_parameters[name].grad + unrolled_parameters['.'.join(parts)].grad
        else:
            reference = unrolled_parameters[name].grad
        torch.testing.assert_close(parameter.grad, reference, atol=1e-6, rtol=1e-5)
    loop.eval()
    changed = x.clone()
    changed[:, 7:] = torch.randint(0, 336, (3, 4))
    with torch.no_grad():
        torch.testing.assert_close(loop(x)[:, :7], loop(changed)[:, :7], atol=1e-6, rtol=1e-6)
    data_root = (Path(__file__).resolve().parents[2] / 'experiments/toy_hybrid/datasets')
    checks = {}
    for seed in range(101, 106):
        path = data_root / f'seed_{seed}.npz'
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert metadata['training_examples'] == 405000
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == metadata['sha256'], seed
        checks[seed] = digest
    result = dict(status='passed', parameters=3331920, unique_layers=4, loops=2, effective_layers=8,
                  shared_parameter_visit_counts=visits, gradient_sum_equivalence='passed',
                  causal_isolation='passed', common_initialization='passed', dataset_sha256=checks)
    (Path(__file__).parent / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
