---
layout: interview-drill
title: ML-coding foundation mock
permalink: /interview-prep/drills/ml-coding-mock-foundation.html
---

# ML-coding foundation mock

Open this file only when the 45-minute timer starts. Keep the check sheet closed.

## Prompt

Implement numerically stable multiclass cross-entropy from logits in PyTorch without calling `cross_entropy`, `log_softmax`, or `softmax`:

```python
def stable_cross_entropy(
    logits: torch.Tensor,   # [..., C]
    targets: torch.Tensor,  # [...]
) -> torch.Tensor:
    """Return the mean negative log-likelihood."""
    ...
```

Requirements:

1. Support any number of leading batch/token dimensions.
2. Use a numerically stable log-sum-exp computation.
3. Gather the target-class logit without one-hot expansion.
4. Preserve gradients, dtype, and device.
5. Reject incompatible shapes, non-integer targets, and out-of-range class indices.
6. Do not use Python loops over examples or classes.

## Interview follow-ups

Answer aloud:

1. Why does subtracting the maximum not change the softmax?
2. What are the shapes before and after `gather`?
3. What happens with very negative logits in float16?
4. How would label smoothing change the expression?
5. What is the time and auxiliary-memory complexity?

## Required tests

- agreement with `torch.nn.functional.cross_entropy` on a fixed random tensor;
- logits containing values near `+1000` and `-1000`;
- a `[B, T, C]` input;
- a backward pass with finite gradients;
- one invalid-target case.

Stop coding at 35 minutes. Use the check sheet during the last 10 minutes.
