---
layout: interview-drill
title: G3b breadth gate answer cues
permalink: /interview-prep/drills/g3b-breadth-gate-answer.html
---

# G3b breadth gate: answer cues

These are checking cues, not scripts to memorize.

1. Absolute/fixed, learned, relative/rotary; discuss length extrapolation and parameterization.
2. Data, tensor, pipeline, sequence/context, and expert parallelism; name split and communication.
3. PPO uses a learned value baseline and clipped ratios; GRPO uses group-relative rewards and can omit the critic.
4. Bipartite matching trains a set prediction with one prediction assigned to each target.
5. Epsilon predicts noise; v is a signal/noise-weighted parameterization with different timestep conditioning.
6. Joint: quadratic in \(TS\). Factorized: \(T S^2 + S T^2\).
7. APH weights detection quality by heading accuracy.
8. Conditional and unconditional predictions are evaluated together.
9. Explicit per-device computation and communication under a named mesh.
10. Prefill has large matrix-matrix work; token-by-token decode repeatedly reads weights and KV state.
11. Regress the conditional vector field along the chosen probability path.
12. Example: static or low-motion video can score well on smoothness.
13. Without balancing, routing collapses onto a few experts and wastes capacity.
14. Linear in batch, sequence, layers, KV heads, head dimension, and bytes per element.
15. A semantically matching non-paired sample is treated as a negative because pairing defines the labels.
