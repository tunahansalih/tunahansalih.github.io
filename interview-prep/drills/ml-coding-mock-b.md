---
layout: interview-drill
title: ML-coding mock B
permalink: /interview-prep/drills/ml-coding-mock-b.html
---

# ML-coding mock B

Open this file only when the onsite-simulation timer starts. Keep the check sheet closed.

## Prompt

Implement one vectorized update step for batched k-means in PyTorch:

```python
def batched_kmeans_step(
    points: torch.Tensor,     # [B, N, D]
    centroids: torch.Tensor,  # [B, K, D]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return new_centroids [B,K,D], assignments [B,N]."""
    ...
```

Requirements:

1. Assign every point to its nearest centroid using squared Euclidean distance.
2. Do not loop over `B`, `N`, or `K`.
3. Update centroids as the mean of assigned points.
4. If a cluster is empty, keep its previous centroid.
5. Preserve dtype and device.
6. Reject incompatible shapes with a clear error.

## Interview follow-ups

Answer these aloud:

1. State every intermediate tensor shape.
2. What is the time and memory complexity?
3. How would you reduce peak memory when `N×K` is too large?
4. Why can the objective fail to decrease in a buggy empty-cluster implementation?
5. How would you make initialization deterministic?

## Required tests

Write and run:

- two obvious 2D clusters with known assignments;
- an empty-cluster case;
- batch size 2 with different cluster layouts;
- CPU and, if available, accelerator device preservation.

Stop coding at 35 minutes. Use the check sheet during the last 10 minutes.
