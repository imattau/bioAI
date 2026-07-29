# Ecosystem Scaling Analysis

**Date:** 2026-07-29
**Experiment:** `experiments/ecosystem_scaling.py`
**Spec:** `checkpoints/scaling_spec.json` (97 lessons, 50 holdout subjects)
**Sample:** 20 holdout subjects, 40 ecosystem evaluations

## Results

| Evidence Size | Recall | Supported Count | Unsupported Rate | Contradiction Rate |
|:---:|:---:|:---:|:---:|:---:|
| 3 (all target) | 0.50 | 1.5 / 3.0 | 0.10 | 0.30 |
| 10 (3 target + 7 distractors) | 0.13 | 0.4 / 3.0 | 0.78 | 0.65 |

## Finding

**Generation quality does NOT improve with library size — it degrades.**

The ecosystem lacks a relevance-gating step. When given 10 evidence items (3 about the target subject + 7 distractors), it treats all evidence equally:
- `extract_propositions()` extracts facts from ALL evidence indiscriminately
- `FixedSpliceCandidateGenerator` creates splice candidates from ALL evidence
- The proposition-track mutation/recombination operators merge facts across unrelated subjects
- The result: responses that mix facts about different subjects, causing contradictions and unsupported assertions

At size 3 (all target facts, no distractors), recall is 0.50 — the ecosystem composes 1.5 of 3 target propositions on average. This shows composition itself works but isn't perfect even in ideal conditions.

## Root Cause

The bottleneck is relevance filtering. The `FixedSpliceCandidateGenerator` caps at 24 candidates regardless of evidence count. But the proposition track (`seed_proposition_population`) creates one organism per source, each containing all propositions from that source. With 10 sources (3 target + 7 distractors), the population has 10 proposition organisms. The selection/recombination rounds operate on all of them, with only token-overlap relevance scoring against the prompt as the selection pressure — and "What do you know about X?" doesn't produce enough overlap to strongly favor X's facts over unrelated ones.

## Implications

1. **Library size alone doesn't help** — the ecosystem needs a relevance filter to select which evidence to compose from
2. **The `FixedSpliceCandidateGenerator`'s 24-candidate cap** is not the bottleneck; the bottleneck is the lack of evidence relevance weighting before proposition extraction
3. **The current ecosystem is useful only when evidence is pre-filtered** (as in `ecology_benchmark.py` where both evidence items are always about the target)
