"""Second two-bug debugging drill.

Keep the debugging drill 02 answer page closed until both checks pass or 40 minutes expire.
Write hypotheses before edits.
"""

import torch
import torch.nn.functional as F


def masked_attention(q, k, v, allowed):
    scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    scores = scores.masked_fill(~allowed, 0.0)
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


def classification_loss(logits, target):
    probabilities = torch.softmax(logits, dim=-1)
    return F.cross_entropy(probabilities, target)


def main():
    torch.manual_seed(11)
    q = torch.randn(1, 1, 3, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)
    allowed = torch.tril(torch.ones(3, 3, dtype=torch.bool))[None, None]

    _, weights = masked_attention(q, k, v, allowed)
    assert torch.all(
        weights.masked_select(~allowed) == 0
    ), "Disallowed attention weights must be exactly zero."

    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
    target = torch.tensor([0, 1])
    actual = classification_loss(logits, target)
    expected = F.cross_entropy(logits, target)
    torch.testing.assert_close(actual, expected)
    print("Both checks pass.")


if __name__ == "__main__":
    main()
