# ML-coding foundation mock — check sheet

Open only after 35 minutes.

## Functional checks

- Validates `targets.shape == logits.shape[:-1]`.
- Computes `m = logits.max(dim=-1, keepdim=True).values`.
- Computes log-sum-exp from shifted logits, then restores the maximum.
- Gathers target logits with a trailing singleton index and removes that dimension.
- Returns the mean of `logsumexp - target_logit`.
- Does not detach or move tensors.
- Invalid dtype/range/shape checks are clear.

## Reasoning checks

- Subtracting the same scalar from every class cancels in the softmax numerator and denominator.
- `targets.unsqueeze(-1)` has shape `[...,1]`; gathered logits become `[...,1]` before squeezing.
- Stable shifting prevents exponent overflow, although very small exponentials may underflow in low precision.
- Time is `O(number of logits)`; the shifted/exponential tensor is the main auxiliary allocation.

## Score

Score 0–2 on each:

1. decomposition and narration;
2. numerical stability;
3. general shape handling;
4. validation and tests;
5. recovery from mistakes.

Pass: at least 8/10, no zero, and numerical agreement with the reference.
