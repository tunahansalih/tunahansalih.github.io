---
layout: interview-drill
title: ML-coding mock A check sheet
permalink: /interview-prep/drills/ml-coding-mock-a-check.html
---

# ML-coding mock A: check sheet

Open only after 35 minutes.

## Functional checks

- Both embedding tensors are normalized along the last axis.
- `logits_per_image` has shape `[B, B]`.
- `logits_per_text` is the transpose of the image logits.
- Targets are `arange(B)` on the correct device.
- Both cross-entropies are computed and averaged.
- No tensor is detached and gradients reach `logit_scale`.
- Dimension and batch mismatches fail clearly.

## Reasoning checks

- Time complexity: `O(B²D)`.
- Logit memory: `O(B²)`.
- Symmetry trains both retrieval directions.
- Unbounded scale can create extreme logits and unstable/overconfident optimization; production implementations commonly constrain or regularize it.
- Multiple positives require a multi-positive target or masked log-sum-exp formulation rather than one class index.

## Score

Score 0–2 on each:

1. decomposition before coding;
2. tensor-shape correctness;
3. numerical/gradient correctness;
4. tests and edge cases;
5. narration and recovery.

Pass: at least 8/10, no zero, and the core loss runs.
