---
layout: interview-drill
title: ML-coding mock B check sheet
permalink: /interview-prep/drills/ml-coding-mock-b-check.html
---

# ML-coding mock B: check sheet

Open only after 35 minutes.

## Functional checks

- Pairwise squared distances have shape `[B, N, K]`.
- Assignments use `argmin` over `K` and have shape `[B, N]`.
- Point sums and counts are accumulated per batch and cluster without loops.
- Counts are clamped only for division; a separate empty mask restores old centroids.
- Output shapes, dtype, and device match the contract.
- Shape checks cover batch and feature dimensions.

## Reasoning checks

- Time complexity: `O(BNKD)`.
- Materialized distance memory: `O(BNK)`.
- Chunking over `N` or `K` reduces peak distance memory.
- Replacing an empty centroid with zeros or NaNs can increase the objective and corrupt later assignments.
- A fixed seed plus a defined initializer, such as fixed-index samples or deterministic k-means++, makes initialization reproducible.

## Score

Score 0–2 on each:

1. decomposition before coding;
2. vectorization and shape correctness;
3. empty-cluster correctness;
4. tests and performance reasoning;
5. narration and recovery.

Pass: at least 8/10, no zero, and the core step runs.
