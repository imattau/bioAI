# Architecture Status: Design vs. What's Actually Built

`ARCHITECTURE.md` states the design intent for a five-subsystem,
biologically-grounded alternative to transformer/backprop architectures.
This document is a grounded status check against the actual codebase — what
each subsystem's code actually does today, where implementation matches the
design, where it's diverged, and where gaps are confirmed (empirically, not
theoretically) rather than merely suspected. Written 2026-07-28, on branch
`agent/vsa-encoder-generator`, after an extended session building out
`src/vsa/relational.py` and wiring it into `src/text/agent.py` (see
`RELATIONAL_MEMORY.md` for the detailed working log of that subsystem).

**Updated same day** after a cleanup pass (commit `748498e`) removed code
that this status check had identified as unwired: the generation subsystem
(`src/nca/*`, `src/decoder/{dit,diffusion,sampler,decoder}.py`) and the
orphaned `NegativeSelectionDetector`/`DetectorEnsemble`. Sections 3 and 5
below now describe the codebase after that removal, not before it — a
prior version of this document described those components as present but
unwired; they are no longer present at all. See that commit message for
the full audit (each removed component confirmed via grep to have zero
callers outside its own test file, before deletion).

## The core thesis

Transformers trained by backprop have specific, structural failure modes:
no clean way to add knowledge without risking old knowledge, no internal
"I don't know this" signal, binary rather than graduated novelty response,
no runtime self-consistency check, relational reasoning tied specifically to
attention. The bet is that these aren't fixable by patching the transformer
— they need a different substrate. The method: find the biological system
that already solves each functional requirement, rather than starting from
a biological metaphor and looking for a use. What ties the five subsystems
together isn't a shared algorithm, it's a shared computational primitive:
**local, content-addressable, non-destructive memory**, instead of a global
loss function computed over the whole system.

## The five subsystems: design vs. actual code

### 1. Encoding substrate (`ARCHITECTURE.md` §3.1 — spiking networks)

**Design**: event-driven spiking neural network, local learning rules
(STDP, equilibrium propagation), removing backprop's "locking" problem.

**Actual**: no evidence of real spiking dynamics in the codebase. What's
actually implemented for encoding is VSA-based (`src/vsa/primitives.py`,
`src/text/encoder.py`) — bind/bundle operations on high-dimensional vectors,
not spike timing. `ARCHITECTURE.md` §5 already flags this as unproven at
LLM scale; in practice the project substituted VSA encoding for the
spiking-substrate encoding layer rather than building real spiking dynamics.

### 2. Relational memory (§3.2 — hippocampal-style binding)

**Design**: pattern separation (dentate gyrus) + pattern completion (CA3
recurrent attractor dynamics) + grid-cell structured coordinates, via VSA
binding/bundling, as an alternative to self-attention.

**Actual**: the most deeply verified subsystem in this codebase (see
`RELATIONAL_MEMORY.md` for the full working log). `src/vsa/relational.py`
implements this faithfully — partitioned Hopfield nets per relation
(pattern separation, applied literally rather than left as a metaphor),
`complete_detailed`/`resolve()` (pattern completion, including genuine
multi-step evidence intersection), grid-cell structure exists separately
(`src/vsa/grid_cells.py`). Confirmed, not just designed:

- Retrieval crosstalk (too many stored patterns interfering) is fixed by
  partitioning — 100% accuracy on uniquely-determined queries, up from
  ~55-58%.
- Genuine ambiguity (the query really does have multiple valid answers) is
  not fixable by better encoding — it's surfaced honestly instead of
  guessed, which measurably improved outcomes over forcing a single answer.
- Evidence-count weighting and transitive multi-hop chaining are confirmed,
  real gaps — not built yet, and the knowledge-graph-embedding literature
  says this is a general limitation of this class of representation, not
  specific to this code (TransE-style models are "strong single-hop
  predictors but offer no native mechanism for composing unseen relation
  chains").
- Multi-word entity values needed compositional (bag-of-words-style)
  encoding, not atomic symbols — found via an actual multi-turn
  conversation, not a unit test: "a mammal" and "mammal" were unrelated
  random vectors before this was fixed, a silent miss rather than even an
  ambiguity.

### 3. Continual learning / self-monitoring (§3.3 — immune-style)

**Design**: clonal selection (clone + locally mutate on novel input,
non-destructive) and negative selection (detect internal states that don't
match a learned "self" baseline).

**Actual**: `src/clonal/` (`ClonalModule`, `ClonalPool`) implements cloning
on novel input — cosine-affinity-threshold matching, Gaussian-noise
mutation, and a per-module local gradient update (not backprop across the
whole network), wired into `BioAIDialogueAgent._detect_novelty`.
`src/immune/` now contains exactly one mechanism: `SelfMonitor` (Hopfield
energy-based drift detection), which is what `_detect_novelty` actually
calls — recall a query through the Hopfield net, score its energy
z-score against a calibrated baseline, and treat high-z as anomalous
enough to trigger `ClonalPool.process()`.

A `NegativeSelectionDetector`/`DetectorEnsemble` (one-class-SVM-based,
matching this section's own suggested implementation almost verbatim)
used to exist in `detector.py`, but was removed in a 2026-07-28 cleanup
after a full-repo grep audit found it had zero callers anywhere — not
`agent.py`, not any experiment script, only its own test file. It wasn't
a case of "built but not yet wired in" the way (4)'s `self.gonogo` was
before this session — it was never called by anything. `SelfMonitor`
already covers this section's actual functional claim (§4: "negative
selection computes its check as a pattern-completion query... using the
same machinery as ordinary retrieval") more directly than the removed
sklearn detector would have, since it reuses the Hopfield net directly
rather than a separate model. Danger-theory-style contextual gating
(§3.3's "gate on harm signals, not just novelty") remains unimplemented —
`_detect_novelty` gates on energy anomaly alone.

### 4. Action selection (§3.4 — basal ganglia)

**Design**: Go/NoGo actor-critic with dopaminergic reward-prediction-error,
generalized to "which memory to attend to" as an action-selection problem,
not just motor actions.

**Actual (updated since first written)**: `src/basal/actor_critic.py`
(`GoNoGoActorCritic`) is now wired into two places, with different
maturity. (1) `src/basal/relevance.py` (`RelevanceSelector`) wraps it for
`RelationalMemory.resolve_auto`'s query-chain selection (`RELATIONAL_MEMORY.md`
§2.7-2.8) — a real "which memory to attend to" role with a genuine
self-supervised reward (whether a chosen query narrowed the candidate
pool, read directly from `resolve()`'s own ground-truth-checked
`informative` flag). (2) `BioAIDialogueAgent.gonogo` itself — previously
created in `__init__` but never called anywhere, genuinely dead in the
live pipeline — now makes a real Go/NoGo decision on every general-retrieval
question turn (`_retrieve_context`), visible as `gonogo_go`/`gonogo_action`
on the result. Its behavioral influence is gated off by default
(`gonogo_gate_enabled`) and it only learns from explicit `record_feedback`
calls, because unlike (1), there's no ambient ground-truth signal for
"was this general free-text retrieval actually correct" — the agent can't
determine that on its own, so nothing pretends to learn from ordinary
conversation without real external supervision.

`tests/test_basal/test_odyssey_arena.py`'s bandit-style task remains a
separate, simpler validation of the same underlying actor-critic
mechanics.

**`experiments/gonogo_feedback_benchmark.py`** is that external caller,
and it found two real bugs in the (2) wiring by actually training it,
not just calling it once:

1. gonogo's state was the raw encoded *query text* — unique per question,
   carrying no information about *retrieval quality*. Training on one
   question's outcome had no way to transfer to a different question, so
   agreement with ground truth got *worse* over a run (60% early → 40%
   late), not better. Fixed: state is now retrieval quality (score,
   margin) embedded via two fixed random axes, decoupled from question
   content — the correct inductive bias, since "should I trust this
   match" depends on match quality, not topic.
2. `GoNoGoActorCritic.act()` has no exploration floor (unlike
   `RelevanceSelector`, which needed one for exactly this reason — see
   `RELATIONAL_MEMORY.md` §2.7), so it can get permanently stuck favoring
   one action from an unlucky initialization. Fixed with the same
   epsilon-greedy floor.

Honest result after both fixes: learning is real but not yet reliably
convergent at the tested sample sizes (~40-60 LLM-generated scenarios) —
runs show improving agreement (e.g. 23%→51%, 58%→69%) as often as flat or
noisy ones, likely some combination of small sample, a genuinely noisy
"correct" label (imperfect substring-match ground truth), and no further
hyperparameter tuning attempted. Both fixes were necessary and are
confirmed correct in isolation (regression tests in
`tests/test_text/test_gonogo_wiring.py`); full convergence at practical
scenario counts is an open question, not yet resolved.

**`record_feedback` is now also wired into the live conversation loop
itself**, not just external benchmark callers: `process_turn` detects a
short confirmation ("yes", "that's correct") or correction ("no", "that's
wrong") directly following a general-retrieval question turn
(`_detect_feedback_signal`) and calls `record_feedback` automatically,
using the user's own next message as the reward — the one source of
feedback in ordinary conversation that isn't fabricated, unlike trying to
invent a proxy signal from the retrieval's own internals. Bounded
deliberately: only applies when the feedback-shaped message is the
*immediately* following turn (an intervening unrelated turn breaks the
link, since a later "yes" is ambiguous about what it confirms), and the
detector is a first-word-plus-length heuristic, not real natural-language
understanding — a short reply starting with "no" for an unrelated reason
right after such an answer (e.g. "No thanks") could be misread as a
correction. That's a real, disclosed limitation of a heuristic detector,
not a hidden one. The behavioral gate (`gonogo_gate_enabled`) is still off
by default regardless — this wiring makes gonogo *learn* continuously from
real conversation, it doesn't by itself make gonogo's decisions start
mattering.

### 5. Generation (§3.5 — developmental/NCA)

**Design**: Neural Cellular Automata, coarse-to-fine unfolding conditioned
on the VSA memory store, preferred specifically because it's non-attention,
non-token-parallel, structurally compatible with the rest of the
architecture.

**Actual**: nothing. As of the 2026-07-28 cleanup, `src/nca/` and
`src/decoder/{dit,diffusion,sampler,decoder}.py` no longer exist in this
codebase. Requirement 7 (extended generation) has no implementation at
all right now — every response `BioAIDialogueAgent` produces is either a
verbatim retrieved sentence or a fixed template string ("I'll remember
that."), never a generated one.

What used to be there, and why it was removed:

- `src/decoder/dit.py`'s `VSAConditionedDiT` (a diffusion transformer
  conditioned on VSA vectors) was a genuine divergence from the design's
  own stated preference for NCA specifically *because* it avoids
  attention — DiT internally uses self-attention. Its own training
  script (`experiments/train_decoder.py`) documented the outcome plainly:
  near-zero training loss but only ~6% exact-match accuracy (see
  `checkpoints/decoder-training-autopsy.md`), a worse and more expensive
  result than deterministic exact-match retrieval. Never called by
  `agent.py`.
- `src/nca/` (`NCACell`, `NCA`, `CoarseConditioner`) matched the *shape*
  of a Growing Neural Cellular Automata API (a cell, a step loop, a
  coarse conditioner projecting a VSA vector to a grid) but not its
  substance. Checked against the actual research (Mordvintsev et al.,
  "Growing Neural Cellular Automata," Distill 2020): the implementation
  had the stochastic per-cell update (`fire_rate` masking) correct, but
  used learned 3×3 convs where the paper uses fixed Sobel-filter
  perception, had no alive-masking/cell-death mechanism (the actual
  growth mechanism — without it this is a recurrent filter over a
  fixed-size grid, not something that grows from a seed), and had no
  training loop at all, let alone the damage/persistence sample-pool
  curriculum that produces the paper's headline self-repair behavior.
  It was never trained, so it could not have exhibited growth or
  self-repair even in principle. `maze_data.py`'s synthetic target
  shapes (checkerboard, stripes, etc.) were never consumed by any
  training script — only by that module's own now-removed test.

`LookupDecoder` (`src/decoder/lookup_decoder.py`) is the only remaining
component in `src/decoder/`, and it isn't a generator either — it's
deterministic Hopfield-cleaned exact-match retrieval over ingested
sentences, the same category of mechanism as `RelationalMemory` and
`ConsolidationMemory`, not an answer to requirement 7.

## How the pieces actually connect

`ARCHITECTURE.md` §4 claims pattern completion, clonal memory, and negative
selection all operate as queries/insertions into the same associative-store
primitive. This holds for the pieces directly touched this session:
`SelfMonitor` queries `HopfieldNet` directly; `ConsolidationMemory` and
`RelationalMemory` are both, structurally, pattern-completion-based
associative stores wired into the same dialogue agent.

**A live seam the design doc doesn't call out**: `BioAIDialogueAgent` runs
*two independent, unmerged* relational reasoners in parallel —
`ConsolidationMemory` (plain-dict, exact-string matching, has evidence
counting and transitive chaining) and `RelationalMemory` (VSA-based, has
ambiguity-surfacing and multi-step candidate intersection). Neither
subsumes the other yet; each covers gaps the other doesn't. This is an
honest current limitation of §3.2's implementation, empirically confirmed
in `tests/test_vsa/test_relational_vs_consolidation.py`, not a design
decision — the natural resolution is closing `RelationalMemory`'s two
confirmed gaps (evidence weighting, chaining) and retiring
`ConsolidationMemory`'s relational half, but that hasn't happened yet.

## Where this sharpens `ARCHITECTURE.md`'s own stated uncertainty

§5 already names the load-bearing uncertainty: "whether pattern completion
can substitute for attention at the scale and flexibility needed for
open-ended reasoning over long, varied context." Everything found this
session is consistent with that framing but more specific: pattern
completion works cleanly when the query is well-posed (100% accuracy in
controlled benchmarks). The actual risk isn't crosstalk or capacity — both
are fixable, and were fixed. It's that natural-language questions are often
*genuinely underdetermined* (multiple truly valid answers) or *phrased
differently from how the fact was originally stored* — problems about the
interface between free text and the symbolic layer, not about the VSA math
itself. Both are now handled (ambiguity is surfaced rather than guessed;
paraphrases are matched via compositional encoding), but both had to be
found by actually building and stress-testing the system, not by reasoning
about the architecture in the abstract.

## Related documents

- `ARCHITECTURE.md` — the original design document (design intent, not
  current-state).
- `RELATIONAL_MEMORY.md` — detailed working log for subsystem 2 (relational
  memory): what's implemented, commit-by-commit, with research citations
  and a prioritized next-steps list.
- `BENCHMARK_SUMMARY.md` — quantitative results across earlier benchmark
  phases (needle-in-haystack retrieval, concept drift, contradiction
  detection, distractor resistance, NCA sequence generation). Its NCA
  section describes code (`src/nca/maze_data.py`) that no longer exists
  as of the 2026-07-28 cleanup above — read it as a historical record of
  that earlier benchmark run, not a description of the current codebase.
- `checkpoints/decoder-training-autopsy.md` — postmortem on the removed
  diffusion decoder (`VSAConditionedDiT`): what was tried, why it
  underperformed `LookupDecoder`, kept as the historical record of that
  decision even though the code it discusses is gone.
