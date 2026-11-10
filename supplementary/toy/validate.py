"""Scientific checks: causal isolation, kernel equivalence, splits and scoring."""
from pathlib import Path
import json
import numpy as np
import torch

from data import make_dataset, OPS, FORBIDDEN, BOS, SEP, QUERY, EOS, PAD
from model import ToyModel, linear_attention, weighted_loss


def validate_model():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    baseline = ToyModel('transformer')
    torch.manual_seed(123)
    hybrid = ToyModel('hybrid')
    for k, v in baseline.state_dict().items():
        assert torch.equal(v, hybrid.state_dict()[k]), k
    for model in [baseline, hybrid]:
        model.eval()
        a = torch.randint(0, 336, (3, 11))
        b = a.clone(); b[:, 7:] = torch.randint(0, 336, (3, 4))
        with torch.no_grad():
            torch.testing.assert_close(model(a)[:, :7], model(b)[:, :7], atol=1e-6, rtol=1e-6)
        logits = model(a)
        loss = weighted_loss(logits, a.roll(-1, 1), torch.ones_like(a).float())
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    q, k, v = [torch.randn(2, 3, 7, 8) for _ in range(3)]
    dense = linear_attention(q, k, v)
    qp, kp = torch.nn.functional.elu(q) + 1, torch.nn.functional.elu(k) + 1
    state = torch.zeros(2, 3, 8, 8)
    norm = torch.zeros(2, 3, 8)
    reference = []
    for t in range(7):
        state += kp[:, :, t, :, None] * v[:, :, t, None, :]
        norm += kp[:, :, t]
        numerator = torch.einsum('bhd,bhdv->bhv', qp[:, :, t], state)
        denominator = (qp[:, :, t] * norm).sum(-1, keepdim=True)
        reference.append(numerator / denominator)
    torch.testing.assert_close(dense, torch.stack(reference, 2), atol=1e-6, rtol=1e-5)
    # A layer-0 attention mask removes the specified directed edge.
    blocked = torch.tensor([[2, 1], [3, 0]])
    got = linear_attention(q, k, v, blocked)
    assert not torch.allclose(got, dense)
    return {k: sum(p.numel() for p in m.parameters()) for k, m in [('transformer', baseline), ('hybrid', hybrid)]}


def validate_dataset(path):
    d = np.load(path)
    assert len(d['train_x']) == 405000
    for train_ids, held_ids in zip(d['train_operand_ids'], d['held_operand_ids']):
        assert len(np.unique(train_ids)) == 90000
        assert len(set(train_ids) & set(held_ids)) == 0
    x, y, w = d['train_x'], d['train_y'], d['train_w']
    seen = set()
    for i in range(270000, len(x)):
        sep = int(np.where(x[i] == SEP)[0][0])
        mem = x[i, 1:sep].tolist()
        assert 3 <= len(mem) <= 6 and len(set(mem)) == len(mem)
        assert x[i, sep + 1] == QUERY
        target = int(x[i, sep + 2]); idx = mem.index(target)
        answer = mem[idx - 1]
        assert idx > 0 and y[i, sep + 2] == answer
        assert not any(p in FORBIDDEN for p in zip(mem, mem[1:]))
        assert (target, answer) not in FORBIDDEN
        sig = (*mem, -1, idx)
        assert sig not in seen; seen.add(sig)
        assert w[i, sep + 2] == 10 and w[i, sep + 1] == 2
        assert np.all(w[i][y[i] == PAD] == 0)
    for i, row in enumerate(d['standard_x']):
        sep = int(np.where(row == SEP)[0][0])
        mem = row[1:sep].tolist(); target = int(row[sep + 2])
        assert (*mem, -1, mem.index(target)) not in seen
    for op in range(3):
        assert len(d[f'semantic_{op}_x']) == 2560
        x = d[f'semantic_{op}_x']; pos = d[f'semantic_{op}_pos']; edges = d[f'semantic_{op}_edge']
        for i, row in enumerate(x):
            bpos, apos = edges[i]
            assert [row[apos], row[bpos]] == list(OPS[op])
            assert row[bpos + 1] == row[pos[i]]
            assert d[f'semantic_{op}_answer'][i] == OPS[op][1]
    return {'dataset': str(path), 'training_examples': len(d['train_x']), 'status': 'passed'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='outputs/toy')
    root = Path(parser.parse_args().root)
    root.mkdir(parents=True, exist_ok=True)
    params = validate_model()
    dataset = make_dataset(101, root / 'datasets')
    result = {'model_checks': 'passed', 'parameters': params, 'dataset_checks': validate_dataset(dataset)}
    (root / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
