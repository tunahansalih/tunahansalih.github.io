"""Two-bug interview drill.

Rules:
1. Before editing, write two hypotheses.
2. State a hypothesis before every edit.
3. Open the debugging drill 01 answer page only after both checks pass or 35 minutes expire.
"""

import torch
import torch.nn.functional as F


def train_step(model, optimizer, x, target):
    logits = model(x).transpose(0, 1)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    optimizer.step()
    return loss


def main():
    torch.manual_seed(7)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    x = torch.randn(8, 4)
    target = torch.randint(0, 2, (8,))

    train_step(model, optimizer, x, target)
    first_grad = model.weight.grad.detach().clone()
    train_step(model, optimizer, x, target)
    second_grad = model.weight.grad.detach().clone()

    assert first_grad.shape == model.weight.shape
    assert torch.allclose(
        first_grad, second_grad, atol=1e-7
    ), "The same batch should not accumulate a second copy of the gradient."
    print("Both checks pass.")


if __name__ == "__main__":
    main()
