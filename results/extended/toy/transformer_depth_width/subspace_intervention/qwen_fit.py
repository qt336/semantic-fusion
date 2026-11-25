"""Original Qwen3-8B linear fit function; standalone PyTorch import."""
import torch

def fit(x, y, valid_x, valid_y, device):
    mean = x.mean(0)
    sd = x.std(0, unbiased=False).clamp_min(1e-6)
    x = ((x-mean)/sd).to(device)
    valid_x = ((valid_x-mean)/sd).to(device)
    y, valid_y = y.to(device), valid_y.to(device)
    classifier = torch.nn.Linear(x.shape[-1], int(y.max())+1, device=device)
    torch.nn.init.zeros_(classifier.weight)
    torch.nn.init.zeros_(classifier.bias)
    opt = torch.optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=0.01)
    best = (-1, -float('inf'))
    rng = torch.Generator().manual_seed(20261125)
    for epoch in range(1, 101):
        for ids in torch.randperm(len(x), generator=rng).split(256):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(classifier(x[ids]), y[ids])
            loss.backward()
            opt.step()
        with torch.no_grad():
            scores = classifier(valid_x)
            key = (int((scores.argmax(1)==valid_y).sum()), -float(torch.nn.functional.cross_entropy(scores, valid_y)))
            if key > best:
                best = key
                saved = dict(epoch=epoch, mean=mean, sd=sd, weight=classifier.weight.detach().cpu().clone(),
                             bias=classifier.bias.detach().cpu().clone())
    return saved
