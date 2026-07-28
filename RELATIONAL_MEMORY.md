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

## 3. What this does and doesn't solve

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

## 4. Next possible steps

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
   `ARCHITECTURE.md` §3.4's own framing.
3. **Persistence for `RelationalMemory`.** Unlike `HopfieldNet` /
   `BioAIDialogueAgent`, `RelationalEncoder`/`RelationalMemory` currently have
   no `get_state`/`from_state` (or `set_state` for the per-relation nets) —
   a save/load round trip would need to serialize `role_vectors`,
   `entity_vectors`, `relation_vectors`, and each relation's `HopfieldNet`
   patterns (see `src/vsa/hopfield.py`'s `set_state`, which itself needed a
   fix this session for exactly this kind of gap).
4. **Secondary partitioning by context, if a relation's net gets large.**
   Not needed for correctness (context binding already reduces crosstalk
   within a relation's net for free), but if a single relation accumulates
   many thousands of triples, partitioning that net further by a
   high-cardinality context role would keep `recall()` cheap. Purely a
   scaling concern, not something today's data justifies.
5. **Self-monitoring hook for contradictions.** `resolve()`'s
   `contradictory` flag and `tests/test_integration/test_contradiction.py`'s
   logic are doing the same kind of check (does new evidence conflict with
   what's already established) in two separate places. `src/immune/monitor.py`
   is the designated home for this kind of self-consistency check per
   `ARCHITECTURE.md` §3.3 — worth unifying rather than leaving contradiction
   detection duplicated ad hoc wherever it's needed.
6. **Confidence combination beyond set intersection**, if pure intersection
   turns out too brittle in practice (e.g. one bad query wipes a pool that
   should have just been down-weighted). Not pursued now because it would
   trade the current design's main property — one clearly-flagged
   contradiction beats a silently blended, falsely-confident score — for a
   softer but less legible signal. Only worth revisiting if `resolve()`
   sees real use and contradiction-flagging proves too strict in practice.
