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

**Update (`commit 1099002`) — ground-truth-exact final answer.**
`final_candidates`/`resolved` now prefer an exact intersection of the
literal stored triples over the per-step top-k VSA intersection above,
whenever every step in the chain has at least one ground-truth match —
the same principle `ambiguous` already applies in `complete_detailed`,
extended to the multi-step case. This was found via genuine
nondeterminism while wiring `resolve()` into real usage (§2.5), not just
theory: `top_k` truncates each step's candidate list, so a real candidate
could be silently dropped from the approximate intersection by chance
even when the exact combination was uniquely recorded in memory. The
ground-truth intersection mirrors the same contradiction-preserving rule
(a conflicting step is skipped rather than collapsing prior progress to
nothing), implemented as `_ground_truth_intersection`.

### 2.5 Persistence, and wiring into `BioAIDialogueAgent` (`commits 1099002`, `f69fd5f`)

**Persistence.** `RelationalEncoder`/`RelationalMemory` gained
`get_state`/`from_state`, closing next-step #3 below. Stored as float16
(matching the convention already used for `VSAEncoder`'s
`word_cache`/`pos_vectors`), and `AssociativeStore` isn't persisted
redundantly — it only backs `exact_lookup` (a diagnostic helper, not used
by `complete_detailed`/`resolve`), so it's rebuilt from `triples` on load
instead of doubling the save size.

**Wiring.** `BioAIDialogueAgent.relational` is fed in parallel with
`ConsolidationMemory` from the same `extract_relations()` output in
`_store_turn`, using its own VSA independent of `vsa_dim` (which is
sometimes set very small, e.g. 64, for the bag-of-words text
encoder/novelty-detection path — reliable VSA bind/unbind needs real
dimensionality). `process_turn`'s question branch tries
`_answer_relational_query` before falling back to ordinary retrieval:

- Single-relation questions ("what is X's Y") use `complete_detailed`
  directly.
- Multi-clue identity questions ("who is X and is Y") use `resolve()` to
  intersect independent evidence about the same unknown subject — the one
  question shape that actually fits `resolve()`'s design. "What is X's Y"
  doesn't fit it at all: there the tied candidates are alternative values
  of *one* (subject, relation) pair, not one entity's membership across
  *several* different relations, so no second query targeting the same
  missing role naturally exists for that shape.
- Genuine ambiguity is surfaced honestly ("it could be X or Y — I don't
  have enough information to be sure") instead of guessed, with
  `ambiguous`/`ambiguous_candidates` now on `process_turn`'s return dict
  and a `relational_reasoning`/`relational_ambiguous` `response_mode`.

**Two correctness issues found by wiring this into real usage, not just
isolated tests** (the whole reason to do this wiring rather than stop at
§2.4 — see also the ground-truth-exact `resolve()` fix above, found the
same way):

- **Provenance must never be bound as VSA context.** A context role
  shares the entity-vector namespace used for decode candidates *by
  design* (§2.3's point about safe cross-role vocabulary sharing) — but
  binding a sentence-id string in as `context` leaked raw id strings into
  subject/object candidate lists. Fixed by tracking source ids as plain
  agent-level metadata (`_relational_sources`) instead of VSA context,
  persisted separately.
- **"Who is X" is genuinely ambiguous** between "describe X" (X is the
  subject) and "which entity has property X" (X is a value some subject
  has) — `ConsolidationMemory.parse_relation_query` always takes the
  former reading. A property phrase like "a mammal" landing there as a
  literal, never-asserted subject would otherwise get a soft Hopfield
  nearest-match guess instead of an honest "I don't know" — the same
  false-confidence failure mode §2.2 exists to prevent, just via a
  different route (a query for a subject that was never stored at all,
  rather than a query with multiple stored matches). Guarded by refusing
  to answer unless the subject was actually ever stored as a subject.

### 2.6 Ground-truth-only candidate display, and compositional entity encoding (`commits 496bcb5`, `6fb835a`)

**Ground-truth-only candidate display (`commit 496bcb5`).** Found via a
longer multi-turn conversation, not a unit test: `complete_detailed`'s
top-k candidates are drawn from the whole shared entity-vector namespace
(every entity ever seen, across every relation), so in a small/young
vocabulary an unrelated low-score entity from a completely different fact
could still make the top-k cut — "What is Pluto?" once listed `"france"`
(from an unrelated capital-city fact) as a plausible answer alongside "a
planet"/"a dwarf planet". Fixed by building `_answer_relational_query`'s
`ambiguous_candidates` list from `ground_truth_ambiguity` (the literal
stored triples) instead of `result.candidates` — ground truth can't
contain noise, by construction.

**Compositional entity encoding.** A second issue surfaced from the same
kind of real-usage testing: multi-word entity/object values (e.g. "a
mammal" extracted from "Cat is a mammal") were encoded as one opaque
atomic symbol — a single random vector keyed on the whole string. That
means "a mammal", "mammal", and "the mammal" were completely unrelated
random vectors with zero similarity, despite meaning the same thing: not
an ambiguity, a *silent miss* — the mirror image of the collision problem
the rest of this document is about. Collisions are "too many things
match, forced to guess"; this was "the two things that obviously match
don't, because the strings differ."

This is a documented, named problem in the literature — Open Information
Extraction produces exactly these uncanonicalized noun phrases (see
[CESI, Vashishth et al. 2018](https://arxiv.org/abs/1902.00172), which
opens with this exact failure mode for extracted (NP, relation, NP)
triples) — and `ConsolidationMemory.extract_relations` is structurally a
small hand-rolled OpenIE extractor, so this isn't a bug specific to this
codebase's regex choices.

Fix: `RelationalEncoder.entity()` now strips leading stopword articles
("a"/"an"/"the") and composes multi-word values by *bundling* their
constituent word vectors (via a shared `token_vectors` cache), rather
than treating the whole string as one atomic symbol — the standard VSA
compositional-semantics approach to phrase representation ("bundle word
vectors for graceful partial similarity" rather than "one random vector
per exact string"), tracing to the same compositionality argument as
[Mikolov et al.'s phrase-vector work](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality).
Measured effect: "a mammal"/"mammal"/"the mammal" now collapse to the
*same* vector (only "mammal" survives stopword stripping); "a dwarf
planet"/"a planet" get partial similarity (~0.7, sharing "planet");
unrelated concepts stay near-orthogonal (<0.1). A fact stored as "Cat is
a mammal" is now retrievable by a query phrased as "mammal" with full
confidence, not a silent miss.

This only changes vector-similarity-based retrieval (`complete_detailed`,
`resolve()`'s VSA fallback) — `ground_truth_ambiguity`'s exact-string
matching is deliberately untouched (it's the "no guessing" authoritative
signal throughout this document, and paraphrase-matching would undermine
that guarantee), and `token_vectors` is persisted alongside
`entity_vectors`/`relation_vectors` so new entities created after a
save/load round trip still compose consistently with pre-existing ones
that share words.

### 2.7 Basal-ganglia relevance selection for `resolve()`'s query chain

`resolve()` (§2.4) deliberately does not search memory for "what else might
help disambiguate this" — it takes a caller-supplied, ordered chain of
queries. That was an explicit deferral, not a permanent decision:
`ARCHITECTURE.md` §3.4 assigns exactly this — "which memory to attend to"
as a competitive-selection problem, the same one solved for motor actions
— to the basal-ganglia subsystem, and `src/basal/actor_critic.py`
(`GoNoGoActorCritic`) already exists for that role but was, until now, only
exercised on bandit-style reward tasks (`tests/test_basal/test_odyssey_arena.py`),
not wired to anything in `RelationalMemory`.

`RelationalMemory.resolve_auto(queries, candidate_queries, max_steps=3,
...)` closes that gap, narrowly: `queries` are mandatory — processed via
`resolve()` in full first, exactly as given, including its own
contradiction detection — and only if those alone leave the pool ambiguous
does a new `RelevanceSelector` (`src/basal/relevance.py`, wrapping
`GoNoGoActorCritic`) choose which of a **bounded pool of already-available
candidate follow-up queries** (not an open-ended memory search — still not
built, and still not what this does) to try next, instead of requiring the
caller to hand a pre-ordered chain or trying all of them. Splitting
mandatory-vs-optional this way matters: evidence the caller already has
(e.g. every clue a user explicitly stated) must always be checked in full,
not skipped just because the pool got resolved early — only genuinely
optional extra evidence goes through selection.

**Mechanics:**
- **State**: the VSA encoding of the mandatory `queries` (bundled together
  if there's more than one) — a fixed-size vector already available for
  free, representing "this kind of disambiguation situation."
- **Actions**: positional, not identity-based — action *i* means "try
  whichever query is currently at position *i* in the caller's
  `candidate_queries` list." The action space size is fixed
  (`relevance_max_candidates`, default 8, per `GoNoGoActorCritic`'s
  fixed-`n_actions` design), so only the first `relevance_max_candidates`
  entries of any candidate list are ever selectable, and invalid positions
  are masked out of the softmax before sampling.
- **Reward**: read directly from `resolve()`'s own per-step signal — a
  chosen query that turns out `informative` gets +1, `contradictory` gets
  -1, merely uninformative gets **-0.2, not 0**.
- **Learning**: online, via a single-step (bandit-style, no `next_state`)
  actor-critic update after every real call. Not pretrained — a fresh
  selector picks close to uniformly at random and only improves with
  repeated use *on the same `RelationalMemory` instance*, since what counts
  as "useful" depends on that memory's actual stored data, not a generic
  prior.

**Why -0.2 instead of a neutral 0 for "uninformative," found empirically
during development, not decided up front**: a neutral reward gives the
actor-critic no gradient pressure to move away from a bad choice it has
already converged to from initialization — once the critic learns to
expect ~0 for the state-action pair it keeps landing on, `advantage`
(`target - value`) collapses toward zero and the policy gradient stalls.
Confirmed directly: with reward=0 for uninformative picks, roughly 1 in 3
random initializations got permanently stuck always picking a useless
candidate, having no reason to ever try the alternative within any
reasonable number of trials. The -0.2 penalty alone reduced but didn't
eliminate this (still occasional stalls); `RelevanceSelector` also has a
fixed epsilon-greedy exploration floor (default 0.15) on top of the
learned policy, guaranteeing every valid action keeps getting sampled
regardless of how confident (and wrong) the policy currently is. This is
standard, well-understood bandit-RL behavior, not a bug specific to this
wiring — and it's honestly not eliminated, only reduced: `resolve_auto`'s
tests assert the aggregate statistical claim ("usually converges well
within a small trial budget, across independent selectors") rather than
"always converges," because the latter would misrepresent what a real
bandit mechanism actually guarantees.

### 2.8 Wiring `resolve_auto` into `BioAIDialogueAgent` (`commit 16084b7`)

`_answer_relational_query`'s multi-clue branch (§2.5) now escalates to
`resolve_auto` when the user's explicitly stated clues alone leave the
pool ambiguous, instead of giving up at "it could be X or Y":

```python
trace = self.relational.resolve(queries, top_k=2)   # explicit clues, always checked in full
if not trace.resolved and trace.final_candidates:
    exclude = set(clues)
    pool = self._gather_candidate_queries(trace.final_candidates, exclude)
    if pool:
        trace = self.relational.resolve_auto(queries, pool, top_k=2, max_steps=3)
```

`_gather_candidate_queries` builds the **bounded** pool this needed
(§2.7's "explicitly not done" item (a), now done, narrowly): other
`(relation, object)` facts already literally recorded in
`self.relational.triples` about entities still in the tied pool, excluding
the exact `(relation, object)` pairs already tried — not the whole
relation name, since two different clues can legitimately share a relation
(e.g. "is a mammal" and "is loyal" are both relation `"is"`; excluding by
relation name alone was an actual bug caught while wiring this in, where
"Dog is loyal" got incorrectly excluded from the pool just for sharing a
relation with the explicit clues). This is still not an open-ended memory
search — only facts about entities *already in the current tied pool*,
capped the same way `resolve_auto` itself caps any candidate list.

Whichever query actually resolved it — the last explicit clue, or an
auto-selected one — is read from `trace.steps[-1].known` for the response
citation, rather than always citing the last explicit clue regardless of
what actually settled it.

**Still not done**: `resolve_auto`'s single-relation counterpart. "What is
X's Y" (§2.5's other branch) still only uses `complete_detailed` — as
established throughout this document, that question shape's tied
candidates are alternative values of *one* relation, not one entity's
membership across several relations, so `resolve()`/`resolve_auto`'s
intersection mechanism doesn't apply to it regardless of selection
strategy.

### 2.9 LLM-driven validation, and a real parsing bug it found (`commit 988ba98`)

Everything in §2.1-2.8 had been validated against hand-crafted examples
only. `experiments/llm_relational_benchmark.py` closes that gap: an LLM
(via Ollama) generates *content* (entities, properties, countries) through
constrained JSON prompts, and sentences are assembled programmatically in
the exact forms `extract_relations` needs — free-form LLM prose almost
never matches those patterns verbatim (confirmed separately:
`experiments/llm_conversation_benchmark.py`'s own output had 0/10
LLM-generated facts trigger the relational path at all). Each scenario
tests confident single-fact retrieval, genuine ambiguity surfacing,
explicit multi-clue resolution, and the §2.8 `resolve_auto` escalation
path, using a fresh agent per scenario to keep measurements isolated.

This surfaced a real bug, not a hypothetical one: `_parse_multi_clue_identity_question`
split purely on the word "and", which can't distinguish "the conjunction
between two clues" from "the word 'and' occurring inside a property's own
text" — an LLM generated the property "loyal and affectionate", which got
wrongly split into two clues ("loyal", "affectionate"), neither of which
exactly matched the stored value, falling through to noisy vector
similarity and landing on a wrong answer. Fixed with a known-value merge
pass: after the naive split, adjacent pieces are greedily re-joined
(longest span first) whenever the merged text exactly matches a value
already recorded under relation `"is"` in `self.relational.triples` — the
same idea as dictionary-based/maximum-munch tokenization or gazetteer-based
named entity recognition, applied to clue boundaries using the actual
stored vocabulary instead of trying to parse grammar. This can only
recognize clues that match something already stored (it can't discover a
genuinely novel multi-word property it's never seen), which is the right
scope: the point is recognizing which previously-asserted facts a question
refers to, not free-form parsing.

Also found and fixed during this validation: the benchmark script itself
had a comparison bug (plain `.lower()` instead of the agent's own
normalization), producing a false-negative "failure" on
`"Mercedes-Benz"` vs. the correctly-stored `"mercedes benz"` — a reminder
that test-harness bugs can look identical to product bugs until you check
which side of the comparison is wrong.

Current result: 100% pass rate across 57 checks (15 LLM-generated
scenarios), with malformed LLM generations (occasional nested-JSON
non-compliance) correctly rejected as generation failures rather than
silently coerced into garbage entity/property names.

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
- **Was not available at all, now closed (§2.5):** persistence — confirmed
  at the time via `hasattr`; `tests/test_vsa/test_relational_vs_consolidation.py::test_persistence_now_available`
  verifies the round trip now that `get_state`/`from_state` exist.

Conclusion: `ConsolidationMemory` cannot be removed yet. Its relational half
is a real candidate for eventual retirement, but only after the two
remaining gaps above (evidence weighting, transitive chaining) are closed —
see §5 for the research pointing at how.

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

Roughly in order of how directly they extend what's already built. Items 1
and 3 are done (§2.5) — kept numbered in place rather than renumbered, since
§5's research citations reference these numbers directly (e.g. "next-step
#2", "next-step #8").

1. ~~Wire `resolve()` into the actual dialogue/generation pipeline~~ —
   **done, §2.5.** `_answer_relational_query` in `src/text/agent.py` now
   uses `resolve()` for multi-clue identity questions and
   `complete_detailed` for single-relation ones, surfacing ambiguity
   honestly instead of guessing.
2. ~~Automatic relevance selection for `resolve()`'s query chain~~ —
   **done, §2.7-2.8.** `resolve_auto`/`RelevanceSelector` pick which of a
   bounded candidate-query pool to try next, learning online from
   reward-prediction-error; `_answer_relational_query` builds that pool
   from other already-recorded facts about the tied candidates
   (`_gather_candidate_queries`, a bounded lookup, not a search) and
   escalates to it when the user's explicit clues alone aren't enough.
   Still open: per §5, this selector is also the natural home for
   transitive-chaining edge-selection (MINERVA-style) once chaining
   (next-step #5) is built, not a separate mechanism — not yet connected
   to that.
3. ~~Persistence for `RelationalMemory`~~ — **done, §2.5.**
   `RelationalEncoder`/`RelationalMemory.get_state`/`from_state` serialize
   `role_vectors`/`entity_vectors`/`relation_vectors` and each relation's
   `HopfieldNet` patterns (float16), and `BioAIDialogueAgent.save`/`load`
   persist `self.relational` and `self._relational_sources`.
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
