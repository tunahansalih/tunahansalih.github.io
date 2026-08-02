# Experiment-design mock

Time: 45 minutes. Ask the follow-ups in order; do not reveal them all at once.

## Main prompt

A new conditioning method claims to improve action controllability in
long-horizon text-to-video generation without reducing visual quality.
Design the experiment that would determine whether the claim is true.

Require the candidate to state assumptions before proposing the design.

## Follow-ups

1. What are the strongest baseline and the deliberately weak baseline?
2. Which confound could make controllability appear better without improving causal response to the action?
3. Define the evaluation unit, dataset split, and at least one long-tail slice.
4. Give one automatic metric, one human-evaluation question, and one known blind spot of each.
5. What result would make you abandon the method?
6. Compute the minimum comparison matrix if there are three methods, two horizons, and four action types.
7. The quality metric improves but human preference falls. What do you do next?
8. The effect disappears on unseen environments. Which hypothesis becomes more likely?
9. You receive only 20% of the requested compute. What is the smallest decisive experiment?
10. Name one leakage check and one reproducibility check.

## Scoring

Score 0-3 on correctness, decomposition, communication, testing/rigour, and
time management. Pass at 11/15 with nothing below 2.
