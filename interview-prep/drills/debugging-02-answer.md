---
layout: interview-drill
title: Debugging drill 02 answer key
permalink: /interview-prep/drills/debugging-02-answer.html
---

# Debugging drill 02: answer key

1. **Masking bug:** replacing disallowed scores with `0.0` still gives them
   nonzero softmax probability. Replace them with negative infinity (or the
   most negative finite value appropriate for the dtype) before softmax.
2. **Loss-contract bug:** `torch.nn.functional.cross_entropy` expects logits,
   not probabilities. Pass `logits` directly. Applying softmax first changes
   the objective and gradients.

For each fix, the desired narration is: suspected contract, observation that
would confirm it, smallest test, then edit.
