# ML-coding mock A

Open this file only when the 45-minute timer starts. Keep the check sheet closed.

## Prompt

Implement a symmetric image–text contrastive loss in PyTorch:

```python
def symmetric_contrastive_loss(
    image_embeddings: torch.Tensor,  # [B, D]
    text_embeddings: torch.Tensor,   # [B, D]
    logit_scale: torch.Tensor,       # scalar tensor
) -> torch.Tensor:
    ...
```

Requirements:

1. L2-normalize each embedding along `D`.
2. Compute every image–text cosine similarity without Python loops.
3. Multiply logits by `logit_scale.exp()`.
4. The positive for image `i` is text `i`.
5. Return the mean of image-to-text and text-to-image cross-entropy.
6. Preserve gradients through embeddings and `logit_scale`.
7. Reject mismatched batch size or embedding dimension with a clear error.

## Interview follow-ups

Answer these aloud while working:

1. What are the shapes of both logit matrices?
2. Why is the loss symmetric?
3. What numerical or optimization problem can an unbounded `logit_scale` cause?
4. How would multiple valid captions per image change the target?
5. What is the time and memory complexity?

## Required tests

Write and run:

- a shape/dtype/device test;
- a perfect-matching case whose loss is lower than a deliberately permuted case;
- a backward-pass test checking finite gradients for all three inputs;
- a batch-size-1 case.

Stop coding at 35 minutes. Use the last 10 minutes with `ml-coding-mock-a-check.md`.
