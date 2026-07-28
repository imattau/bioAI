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

**Actual**: more built than expected. `src/clonal/` (`ClonalModule`,
`ClonalPool`) implements cloning on novel input. `src/immune/` has *two*
separate mechanisms: `SelfMonitor` (Hopfield energy-based drift detection,
directly wired into `BioAIDialogueAgent`'s novelty check in this session's
work) and a separate `NegativeSelectionDetector`/`DetectorEnsemble` in
`detector.py` matching the doc's negative-selection description, not
touched or verified this session. Whether these two pieces are actually
integrated with each other (danger-theory-style contextual gating, per
§3.3's "gate on contextual harm signals, not just pattern mismatch") is
unconfirmed without reading `detector.py`'s call sites more closely.

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

**Actual**: `src/nca/` has `CoarseConditioner` and `cell.py`, matching the
coarse-to-fine description. But `src/decoder/dit.py` also has a
`VSAConditionedDiT` — a diffusion transformer conditioned on VSA vectors.
This is a notable divergence: the design argues NCA is preferred *because*
it avoids attention, but a DiT internally uses self-attention. Not yet
investigated closely enough to know whether this is a pragmatic fallback
(pure NCA text generation not yet good enough) or a parallel approach being
explored alongside NCA — worth reading `dit.py` and its training code
before drawing conclusions.

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
  detection, distractor resistance, NCA sequence generation).
