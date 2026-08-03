---
layout: interview-drill
title: Debugging drill 01 answer key
permalink: /interview-prep/drills/debugging-01-answer.html
---

# Debugging drill 01: answer key

Open this only after both checks pass or the 35-minute cap expires.

1. **Shape bug:** `model(x)` already has shape `(batch, classes) = (8, 2)`. The transpose changes it to `(2, 8)`, but `cross_entropy` expects the first dimension to match the eight targets. Remove `.transpose(0, 1)`.
2. **Training-state bug:** gradients are never cleared. Add `optimizer.zero_grad()` before `loss.backward()`. Without it, the second call accumulates another copy of the same gradient.

The desired debugging behavior is more important than speed: state the suspected failure, name the observation that would support it, and only then edit.
