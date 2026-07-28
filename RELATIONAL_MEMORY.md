# Relational Memory: Encoding Collisions and How They're Managed

Covers `src/vsa/relational.py` — the hippocampal-style binding module from
`ARCHITECTURE.md` §3.2. This document is the working record of the
collision-handling investigation and design decisions; see git history on
`agent/vsa-encoder-generator` for the commits referenced below.

## 1. The problem

Relational triples `(subject, relation, object)` are encoded by binding each
role vector to its filler and bundling the three bound terms. Retrieving a
missing slot from a partial cue (2 of 3 known) is VSA pattern completion via
a Hopfield net. Two different things can make that completion wrong, and
they need different fixes:

1. **Retrieval crosstalk** — a query's similarity to the *correct* stored
   pattern gets swamped by accidental overlap with *other, unrelated* stored
   patterns that happen to share a role/entity vector.
2. **Genuine data ambiguity** — the partial cue really does match more than
   one stored triple (e.g. two different subjects sharing the same
   `(relation, object)` pair). No amount of encoding or retrieval tuning
   fixes this — it's not a representation defect, it's missing information.

Conflating these two was the original failure: a single global Hopfield net
storing every triple gave ~11-15% slot-completion accuracy even at low load,
which looked like a fundamental capacity problem but was actually almost
entirely (1).

## 2. What's implemented

### 2.1 Partitioned storage (fixes crosstalk)

`RelationalMemory` keeps one Hopfield net per relation instead of one global
net (`commit 743888a`). A query with a known relation only competes against
triples stored under that relation. When the relation itself is the missing
slot, `_infer_relation` broadcasts the query across every net and scores
each by how well its recalled state reconstructs the *known* slots.

Result: unique (non-colliding) queries went from ~55-58% to **100%**
accuracy. This is the dentate-gyrus-style pattern-separation idea from
`ARCHITECTURE.md` §3.2, applied literally rather than left as a metaphor.

### 2.2 Ambiguity is surfaced, not hidden (fixes false confidence)

`complete()` still returns a single `(s, r, o)` tuple for backward
compatibility, but `complete_detailed()` returns a `CompletionResult` with:

- `candidates`: top-k matches with scores
- `confidence`: the winning score
- `ambiguous`: True if the match is genuinely uncertain

`ambiguous` is set primarily from `ground_truth_ambiguity(known, context)`,
which checks the **actual stored triples** for how many really match —
this is the authoritative signal, not a score-gap heuristic, because a
sharp (high-beta) Hopfield recall collapses genuine ties to whichever
candidate has marginally higher score from vector noise, which would hide
real ambiguity if score-gap were the only check.

Measured on synthetic data: **100%** accuracy on uniquely-determined
queries, **94-99%** "true answer in top-3" on genuinely ambiguous ones
(vs. ~55-58% forced-single-answer accuracy before). The takeaway: the
representation isn't lossy, forcing a single answer on an inherently
underdetermined query is what was lossy.

### 2.3 Context roles narrow ambiguity (`commit 8a8bc8f`)

Arbitrary context roles (`"scene"`, `"speaker"`, `"time"`, ...) can be bound
into both storage and query without ever being a completion target:

```python
memory.store_triple("cat", "chases", "mouse", context={"scene": "kitchen"})
memory.store_triple("dog", "chases", "mouse", context={"scene": "garden"})

memory.complete_detailed({"relation": "chases", "object": "mouse"})                        # ambiguous: {cat, dog}
memory.complete_detailed({"relation": "chases", "object": "mouse"}, context={"scene": "kitchen"})  # resolved: cat
```

This is the same mechanism as more context narrowing an LLM's next-token
distribution: it only helps when the context actually *discriminates*
between the colliding triples. If both collided under the same context too,
`ground_truth_ambiguity` correctly still reports ambiguity — no amount of
conditioning manufactures information that was never captured.

`RelationalEncoder.role()` creates role vectors lazily (mirroring
`entity()`/`relation()`), so new context role names don't need to be
pre-declared. `TARGET_ROLES` (subject/relation/object) is kept distinct from
arbitrary context role names. `context` defaults to `None` everywhere, so
all pre-existing call sites are unaffected.

### 2.4 Multi-step candidate intersection (`commit f344545`)

`RelationalMemory.resolve()` takes a *sequence* of independent
`(known, context)` queries targeting the same missing role and intersects
their candidate pools:

```python
memory.store_triple("cat", "chases", "mouse")
memory.store_triple("dog", "chases", "mouse")   # collision: {cat, dog}
memory.store_triple("dog", "fears", "water")
memory.store_triple("fox", "fears", "water")    # collision: {dog, fox}

trace = memory.resolve([
    ({"relation": "chases", "object": "mouse"}, None),
    ({"relation": "fears", "object": "water"}, None),
], top_k=2)
# trace.resolved == True, trace.final_candidates == ["dog"]
```

Design decisions, and why:

- **Intersection, not score averaging.** The entity being resolved has to
  be consistent with *every* piece of evidence simultaneously, so the pool
  only ever shrinks. Averaging would let one noisy/irrelevant step outvote
  a decisive one.
- **`informative` per step.** A step that doesn't shrink the pool
  contributed nothing and must not look like part of a chain of reasoning
  that resolved something — the same discipline as `ambiguous` in
  `complete_detailed`, applied to a multi-step process instead of a
  single query. (Ties back to the discussion of LLM chain-of-thought
  sometimes being post-hoc rationalization rather than the actual cause of
  an answer — a step that adds no information doesn't get credit for one.)
- **Contradiction, not silent erasure.** If a step's candidates would
  empty the pool entirely, that's conflicting evidence, not "no answer" —
  the prior pool is kept and the step is flagged `contradictory`. This is
  checked on *every* query in the chain, even ones after the pool has
  already narrowed to one candidate, so a later conflicting clue can't be
  silently ignored just because a "resolution" was already reached.
- **No automatic query selection, deliberately.** `resolve()` takes the
  query chain as given by the caller; it does not search memory for "what
  else might help disambiguate this." Per `ARCHITECTURE.md` §3.4, deciding
  *which* memory to attend to next is the basal-ganglia relevance-selection
  subsystem's job, and §6's suggested build order puts action selection
  after relational reasoning is solid — building an ad hoc relevance
  heuristic here would jump that sequencing.

## 3. Comparison against `ConsolidationMemory` (empirically tested)

`src/text/agent.py`'s actual live relational reasoning today is
`ConsolidationMemory` (`src/text/consolidation.py`), not `RelationalMemory` —
a separate, plain-dict symbolic store fed by regex-based `extract_relations()`,
with its own tie-breaking (`_supported_choice`) and its own transitive
multi-hop mechanism (`reason_path`). It does two unrelated jobs: relation
extraction/reasoning (overlaps with `RelationalMemory`) and concept-prototype
formation for semantic retrieval scoring (no overlap at all, load-bearing in
`_retrieve_context`, not touched by anything in this document).

`tests/test_vsa/test_relational_vs_consolidation.py` mirrors
`ConsolidationMemory`'s exact test scenarios through `RelationalMemory`
instead, to answer "what can be removed" with evidence rather than
intuition. Result:

- **Already replicated, and improved on:** tie/ambiguity abstention (named
  candidates instead of a bare `None`), confident single-fact retrieval.
- **Genuinely missing** (confirmed empirically — both scenarios actually
  flaked pass/fail on a single unseeded draw during development, which is
  itself informative: a real mechanism would be substrate-independent, so
  both are asserted as 30-trial "never reliably correct" statistical claims
  instead of one draw):
  - **Frequency-weighted evidence.** `ConsolidationMemory` explicitly counts
    repeated claims; `RelationalMemory` has no equivalent — storing the same
    triple twice just appends duplicate vectors, and sharp-beta Hopfield
    recall picks whichever single pattern has marginally higher incidental
    similarity, not whichever claim has more evidence.
  - **Transitive multi-hop chaining.** `reason_path` composes
    `capital_of → in` through an intermediate unknown; `resolve()` only
    intersects independent evidence about *one* unknown — there is no
    unknown-chaining mechanism in `RelationalMemory` at all.
- **Not available at all:** persistence (already tracked as next-step #3
  below, confirmed here directly via `hasattr`).

Conclusion: `ConsolidationMemory` cannot be removed yet. Its relational half
is a real candidate for eventual retirement, but only after the two gaps
above are closed — see §5 for the research pointing at how.

## 4. What this does and doesn't solve

**Solved:** retrieval crosstalk (partitioning), false confidence on
ambiguous queries (surfacing top-k + ambiguous flag instead of guessing),
and collisions that *can* be disambiguated by available context or
independent stored facts (context roles, `resolve()`).

**Not solved, and not solvable this way:** genuinely irreducible ambiguity
— when no stored fact or context anywhere distinguishes the candidates.
`resolve()` will correctly report `resolved=False` with the full remaining
pool in that case rather than guessing, which is the intended behavior, not
a bug to fix later.

**Separately unsolved (different problem):** compositional generalization
to entities never seen at storage time measures 0% (`experiments/relational_reasoning.py`,
`task_comp_generalization`). This is expected for a pure associative-memory
approach — there's no mechanism here for generating a meaningful vector for
a filler that was never bound into anything, that's a semantic/embedding
question, not a collision-handling one.

## 5. Research grounding

Literature check performed 2026-07-28 against the specific gaps above,
rather than VSA/HDC in general. Each item ties to a §6 next step.

- **Crosstalk/partitioning has a formal capacity literature.**
  [Capacity Analysis of Vector Symbolic Architectures](https://arxiv.org/abs/2301.10352)
  formally derives how many bound/bundled items a given dimensionality
  tolerates before noise from unrelated superposed vectors swamps the
  signal — the rigorous version of what partitioning (§2.1) fixed
  empirically by reducing the effective N per net.
- **Sharp Hopfield retrieval forcing a single answer is the same
  choice as attention/decoding, not an implementation quirk.**
  [Ramsauer et al., "Hopfield Networks Is All You Need"](https://en.wikipedia.org/wiki/Modern_Hopfield_Network)
  shows the modern/continuous Hopfield update is mathematically equivalent
  to transformer self-attention (both are a softmax-weighted combination
  over stored patterns); [Krotov & Hopfield's dense associative memory](https://arxiv.org/pdf/2202.04557)
  is the underlying capacity result. [Attention as Binding: A
  Vector-Symbolic Perspective on Transformer Reasoning](https://arxiv.org/html/2512.14709v1)
  makes the bind/unbind correspondence explicit. This is the formal backing
  for §4's "generation/decoding is where irreducible ambiguity gets
  collapsed into false confidence, not retrieval" argument.
- **Evidence-count weighting is a known, solved VSA technique** — the
  standard fix is weighted bundling (bundle codevectors with an explicit
  scalar weight — count, confidence, recency — instead of naive duplicate
  storage). The same [capacity/reliability-estimation literature](https://arxiv.org/abs/2301.10352)
  analyzes this case directly. Confirms the gap found in §3 is a
  straightforward, known fix, not a fundamental VSA limitation.
- **Transitive chaining is a known, hard, actively-researched gap for this
  entire class of representation** — not specific to `RelationalMemory`.
  The knowledge-graph embedding literature says this almost verbatim:
  models like TransE are "strong single-hop predictors but offer no native
  mechanism for composing unseen relation chains at test time." The field's
  fixes are explicit path-composition/graph-walking layers on top:
  [Guu et al.'s path composition](https://arxiv.org/pdf/1805.06197) and
  [MINERVA-style reinforcement-learning graph walkers](https://arxiv.org/pdf/1808.10568).
  MINERVA in particular is structurally the same idea as §6 next-step #2
  (basal-ganglia relevance selection) — an RL agent choosing which edge to
  traverse next — so the eventual chaining mechanism should probably be
  that same subsystem applied to edge-choice, not a hand-rolled traversal.
- **Resonator networks are the field's actual answer to pulling several
  simultaneously-bound unknowns out of one vector.**
  [Frady, Kent, Sommer & Olshausen's resonator networks](https://arxiv.org/abs/2007.03748)
  solve the factorization problem `resolve()` currently approximates via
  brute-force sequential querying + set intersection, searching in
  superposition over all candidate combinations at once and converging via
  nonlinear dynamics — outperforms alternating least squares and
  gradient-based methods. A candidate replacement for §6 next-step #8 if
  sequential intersection proves too slow/brittle at scale, not a
  validation of the current approach.
- **VSA-based compositional reasoning works at real benchmark scale when
  paired with a learned confidence estimate, not fixed thresholds.**
  [Hersche et al., Nature Machine Intelligence 2023](https://www.nature.com/articles/s42256-023-00630-8)
  gets 87.7% on RAVEN's progressive matrices by pairing VSA symbolic
  operators with a neural probability estimator, instead of hand-set
  thresholds like this codebase's `ambiguity_margin`/`hopfield_beta`. An
  argument for eventually learning those thresholds rather than tuning them
  by hand, consistent with `ARCHITECTURE.md`'s general preference for local
  learning over fixed hyperparameters.

## 6. Next possible steps

Roughly in order of how directly they extend what's already built:

1. **Wire `resolve()` into the actual dialogue/generation pipeline**
   (`src/text/agent.py`). Right now it's only exercised in isolation and via
   tests — the natural next step is having the agent actually issue a
   follow-up query when `complete_detailed`/`decode` comes back ambiguous,
   using whatever else is already known in the conversation as the second
   `resolve()` step, instead of just picking `complete()`'s single guess.
2. **Automatic relevance selection for `resolve()`'s query chain** — the
   basal-ganglia piece deliberately deferred in §2.4. `src/basal/` already
   has a Go/NoGo actor-critic; the natural extension is treating "which
   other stored relation might disambiguate this" as the same kind of
   competitive selection problem it already solves for actions, per
   `ARCHITECTURE.md` §3.4's own framing. Per §5, this is also the natural
   home for transitive-chaining edge-selection (MINERVA-style), not a
   separate mechanism.
3. **Persistence for `RelationalMemory`.** Unlike `HopfieldNet` /
   `BioAIDialogueAgent`, `RelationalEncoder`/`RelationalMemory` currently have
   no `get_state`/`from_state` (or `set_state` for the per-relation nets) —
   a save/load round trip would need to serialize `role_vectors`,
   `entity_vectors`, `relation_vectors`, and each relation's `HopfieldNet`
   patterns (see `src/vsa/hopfield.py`'s `set_state`, which itself needed a
   fix this session for exactly this kind of gap).
4. **Evidence-count weighting** (§3/§5). Replace naive duplicate-vector
   storage with explicit weighted bundling — a known VSA technique, not a
   research gap — so repeated claims reliably outrank single conflicting
   ones the way `ConsolidationMemory`'s integer counts do today.
5. **Transitive multi-hop chaining** (§3/§5). `resolve()` intersects
   independent evidence about one unknown; it does not chain through a
   sequence of different unknowns the way `reason_path` does. Needs an
   explicit path-composition or graph-walk mechanism — likely the same
   relevance-selection subsystem as next-step #2, per the MINERVA
   correspondence in §5, rather than a bespoke traversal loop.
6. **Secondary partitioning by context, if a relation's net gets large.**
   Not needed for correctness (context binding already reduces crosstalk
   within a relation's net for free), but if a single relation accumulates
   many thousands of triples, partitioning that net further by a
   high-cardinality context role would keep `recall()` cheap. Purely a
   scaling concern, not something today's data justifies.
7. **Self-monitoring hook for contradictions.** `resolve()`'s
   `contradictory` flag and `tests/test_integration/test_contradiction.py`'s
   logic are doing the same kind of check (does new evidence conflict with
   what's already established) in two separate places. `src/immune/monitor.py`
   is the designated home for this kind of self-consistency check per
   `ARCHITECTURE.md` §3.3 — worth unifying rather than leaving contradiction
   detection duplicated ad hoc wherever it's needed.
8. **Confidence combination beyond set intersection**, if pure intersection
   turns out too brittle in practice (e.g. one bad query wipes a pool that
   should have just been down-weighted). Per §5, resonator networks are the
   principled version of this if it's ever needed. Not pursued now because
   it would trade the current design's main property — one clearly-flagged
   contradiction beats a silently blended, falsely-confident score — for a
   softer but less legible signal. Only worth revisiting if `resolve()` sees
   real use and contradiction-flagging proves too strict in practice.
