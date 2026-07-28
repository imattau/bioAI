"""VSA relational encoding and pattern completion via Hopfield net.

Core architecture prototype (ARCHITECTURE.md Section 6):
  encode triple → Hopfield store (Hebbian) → partial query → Hopfield recall → decode filler

No backprop, no attention, no language. Pure VSA binding + pattern completion.

Storage is partitioned one Hopfield net per relation (dentate-gyrus-style
pattern separation), rather than one global net for every triple. A single
shared net makes every stored triple crosstalk with every query through
shared role/entity vectors, which collapses slot-completion accuracy to
~10-15% even at low load (see experiments/relational_reasoning.py history).
Partitioning by the known relation — or, when the relation itself is the
missing slot, broadcasting the query across all per-relation nets and
scoring each by how well it reconstructs the *known* slots — removes most
of that crosstalk.

What partitioning cannot fix is genuine data-level ambiguity: a partial
cue that really does match more than one stored triple (e.g. two different
subjects sharing the same (relation, object) pair). `complete_detailed`
surfaces this as `ambiguous=True` with the competing candidates, rather
than silently returning one guess.

Context roles (e.g. "scene", "speaker", "time") narrow that ambiguity the
same way more preceding tokens narrow an LLM's next-token distribution:
they're extra bound terms in the query, always supplied (never a slot to
solve for), that only help to the extent they actually discriminate
between the tied candidates. If two colliding triples share the same
context too, context does nothing — no amount of conditioning manufactures
information that was never captured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.vsa.hopfield import HopfieldNet
from src.vsa.primitives import VSA
from src.vsa.store import AssociativeStore


# Roles `complete()` can solve for. Context roles (e.g. "scene", "speaker")
# are arbitrary additional role names supplied alongside `known`/`context` —
# they get their own role vectors lazily (see RelationalEncoder.role) but are
# never a target of completion.
TARGET_ROLES = ("subject", "relation", "object")
ROLE_NAMES = TARGET_ROLES  # backward-compatible alias

# Minimum gap between the top-1 and top-2 candidate scores below which a
# completion is flagged as ambiguous rather than confidently decoded.
DEFAULT_AMBIGUITY_MARGIN = 0.1


class RelationalEncoder:
    """Encode relational triples (plus optional context roles) as VSA vectors."""

    def __init__(self, vsa: VSA):
        self.vsa = vsa
        self.role_vectors: dict[str, torch.Tensor] = {}
        self.entity_vectors: dict[str, torch.Tensor] = {}
        self.relation_vectors: dict[str, torch.Tensor] = {}
        for name in TARGET_ROLES:
            self.role(name)

    def role(self, name: str) -> torch.Tensor:
        """Lazily create the role vector for *name* (target or context role)."""
        if name not in self.role_vectors:
            self.role_vectors[name] = self.vsa.make_vector()
        return self.role_vectors[name]

    def entity(self, name: str) -> torch.Tensor:
        if name not in self.entity_vectors:
            self.entity_vectors[name] = self.vsa.make_vector()
        return self.entity_vectors[name]

    def relation(self, name: str) -> torch.Tensor:
        if name not in self.relation_vectors:
            self.relation_vectors[name] = self.vsa.make_vector()
        return self.relation_vectors[name]

    def filler(self, role: str, name: str) -> torch.Tensor:
        """Filler vector for *name* under *role*. Context roles share the
        entity cache — the same underlying value (e.g. "kitchen") bound
        under different roles (e.g. "scene" vs "object") yields different
        composite vectors, so sharing vocabulary across roles is safe."""
        return self.relation(name) if role == "relation" else self.entity(name)

    def candidates_for(self, role: str) -> dict[str, torch.Tensor]:
        return self.relation_vectors if role == "relation" else self.entity_vectors

    def encode(self, fields: dict[str, str]) -> torch.Tensor:
        """Bind and bundle an arbitrary set of (role, value) pairs.

        Used for both full triples and partial queries — a "query" is just
        a triple encoding with some (target or context) roles omitted.
        """
        if not fields:
            return torch.zeros(self.vsa.dim)
        components = [
            self.vsa.bind(self.role(role), self.filler(role, value))
            for role, value in fields.items()
        ]
        return self.vsa.bundle(components)

    def encode_triple(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: dict[str, str] | None = None,
    ) -> torch.Tensor:
        """Encode (subject, relation, object) [+ context roles] as one bundled vector.

        encode = bind(role_subject, vec(subject))
               + bind(role_relation, vec(relation))
               + bind(role_object, vec(object))
               + sum(bind(role_c, vec(v)) for c, v in context)
        """
        fields = {"subject": subject, "relation": relation, "object": object_}
        fields.update(context or {})
        return self.encode(fields)

    def decode_filler_topk(
        self, state: torch.Tensor, role: str, candidates: dict[str, torch.Tensor], k: int = 3
    ) -> list[tuple[str, float]]:
        """Unbind *role* from *state*, return the top-k closest candidates, best first."""
        unbound = self.vsa.unbind(state, self.role(role))
        names = list(candidates.keys())
        vectors = torch.stack([candidates[n] for n in names])
        sims = self.vsa.similarity(unbound, vectors)
        k = min(k, len(names))
        top_vals, top_idxs = sims.topk(k)
        return [(names[i.item()], top_vals[j].item()) for j, i in enumerate(top_idxs)]

    def decode_filler(
        self, state: torch.Tensor, role: str, candidates: dict[str, torch.Tensor]
    ) -> str:
        """Unbind *role* from *state*, return closest candidate name."""
        return self.decode_filler_topk(state, role, candidates, k=1)[0][0]

    def decode_triple(
        self,
        state: torch.Tensor,
    ) -> tuple[str, str, str]:
        """Decode all three target roles from a retrieved state vector."""
        subject = self.decode_filler(state, "subject", self.entity_vectors)
        relation = self.decode_filler(state, "relation", self.relation_vectors)
        object_ = self.decode_filler(state, "object", self.entity_vectors)
        return subject, relation, object_

    def encode_query(
        self, known: dict[str, str], context: dict[str, str] | None = None
    ) -> torch.Tensor:
        """Encode a partial triple from known slots, plus optional context roles.

        known: {"subject": "cat", "relation": "chases"}
               (object is the slot to retrieve)
        context: {"scene": "kitchen"} — always bound in, never a target.
        """
        fields = dict(known)
        fields.update(context or {})
        return self.encode(fields)


@dataclass
class CompletionResult:
    """Result of completing one missing slot of a triple."""

    role: str
    best: str | None
    confidence: float
    candidates: list[tuple[str, float]] = field(default_factory=list)
    ambiguous: bool = False


@dataclass
class ResolutionStep:
    """One query in a multi-step resolution chain, and its effect on the pool."""

    known: dict[str, str]
    context: dict[str, str]
    result: CompletionResult
    survivors: list[str]
    informative: bool
    contradictory: bool = False


@dataclass
class ResolutionTrace:
    """Result of intersecting candidate pools across a chain of independent queries."""

    role: str | None
    steps: list[ResolutionStep] = field(default_factory=list)
    final_candidates: list[str] = field(default_factory=list)
    resolved: bool = False
    contradictory: bool = False


class RelationalMemory:
    """Relational memory partitioned by relation, with Hebbian (Hopfield) storage.

    One Hopfield net per relation stores the triples that use that relation.
    Retrieval when the relation is known queries that net directly. Retrieval
    when the relation itself is the missing slot broadcasts the query across
    every net and scores each candidate relation by how well its recalled
    state reconstructs the *known* slots (subject and/or object) — the net
    whose stored triples best explain the known slots wins.

    Context roles are bound into both storage and query but never partition
    the store further — they reduce crosstalk within a relation's net for
    free (recall must now match every bound component, not just two), so a
    second partition axis isn't needed for correctness, only potentially for
    scale.
    """

    def __init__(
        self,
        encoder: RelationalEncoder,
        dim: int = 10000,
        hopfield_beta: float = 50.0,
        ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    ):
        self.encoder = encoder
        self.dim = dim
        self.hopfield_beta = hopfield_beta
        self.ambiguity_margin = ambiguity_margin
        self.nets: dict[str, HopfieldNet] = {}
        self.store = AssociativeStore(dim)
        self.triples: list[tuple[str, str, str, dict[str, str]]] = []

    def _net_for(self, relation: str) -> HopfieldNet:
        net = self.nets.get(relation)
        if net is None:
            net = HopfieldNet(self.dim, retrieval_mode="modern", modern_beta=self.hopfield_beta)
            self.nets[relation] = net
        return net

    def store_triple(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: dict[str, str] | None = None,
    ):
        vec = self.encoder.encode_triple(subject, relation, object_, context)
        self._net_for(relation).store(vec)
        self.store.insert(vec)
        self.triples.append((subject, relation, object_, dict(context or {})))

    def _infer_relation(
        self, known: dict[str, str], context: dict[str, str], steps: int, top_k: int
    ) -> CompletionResult:
        query = self.encoder.encode_query(known, context)
        scores: list[tuple[str, float]] = []
        for relation, net in self.nets.items():
            if len(net) == 0:
                continue
            recalled = net.recall(query, steps=steps)
            conf = 0.0
            for role, value in known.items():
                unbound = self.encoder.vsa.unbind(recalled, self.encoder.role(role))
                conf += self.encoder.vsa.similarity(unbound, self.encoder.filler(role, value)).item()
            scores.append((relation, conf))

        if not scores:
            return CompletionResult(role="relation", best=None, confidence=0.0)

        scores.sort(key=lambda pair: -pair[1])
        candidates = scores[:top_k]
        best, best_conf = candidates[0]
        ambiguous = len(candidates) > 1 and (candidates[0][1] - candidates[1][1]) < self.ambiguity_margin * len(known)
        return CompletionResult(role="relation", best=best, confidence=best_conf, candidates=candidates, ambiguous=ambiguous)

    def _infer_slot(
        self, role: str, known: dict[str, str], context: dict[str, str], steps: int, top_k: int
    ) -> CompletionResult:
        net = self.nets.get(known.get("relation"))
        if net is None or len(net) == 0:
            return CompletionResult(role=role, best=None, confidence=0.0)

        query = self.encoder.encode_query(known, context)
        recalled = net.recall(query, steps=steps)
        candidates = self.encoder.decode_filler_topk(recalled, role, self.encoder.candidates_for(role), k=top_k)
        best, best_score = candidates[0]
        ambiguous = len(candidates) > 1 and (candidates[0][1] - candidates[1][1]) < self.ambiguity_margin
        return CompletionResult(role=role, best=best, confidence=best_score, candidates=candidates, ambiguous=ambiguous)

    def complete_detailed(
        self,
        known: dict[str, str],
        context: dict[str, str] | None = None,
        steps: int = 10,
        top_k: int = 3,
    ) -> CompletionResult:
        """Complete the one missing target slot of a partial triple.

        `context` roles (e.g. {"scene": "kitchen"}) are always bound into
        the query but are never completion targets themselves — they exist
        purely to narrow which stored triples the query matches, the same
        way more preceding tokens narrow an LLM's next-token distribution.
        They only help to the extent they actually discriminate between
        otherwise-tied candidates (see `ground_truth_ambiguity`).

        `ambiguous` is set from two independent signals: whether the stored
        triples themselves genuinely contain more than one match for
        `known`+`context` (the authoritative signal —
        `ground_truth_ambiguity`), or-ed with a score-gap heuristic on the
        recalled candidates. The ground-truth check matters because a sharp
        (high-beta) Hopfield recall collapses genuine ties to whichever
        candidate has a slightly higher score from vector noise, which would
        otherwise hide real ambiguity.
        """
        context = context or {}
        missing = [role for role in TARGET_ROLES if role not in known]
        if len(missing) != 1:
            raise ValueError(f"Expected exactly 2 known target slots, got {list(known.keys())}")
        role = missing[0]
        if role == "relation":
            result = self._infer_relation(known, context, steps=steps, top_k=top_k)
        else:
            result = self._infer_slot(role, known, context, steps=steps, top_k=top_k)
        if len(self.ground_truth_ambiguity(known, context)) > 1:
            result.ambiguous = True
        return result

    def complete(
        self,
        known: dict[str, str],
        context: dict[str, str] | None = None,
        steps: int = 10,
    ) -> tuple[str, str, str]:
        """Complete a partial triple via Hopfield pattern completion."""
        result = self.complete_detailed(known, context=context, steps=steps)
        full = dict(known)
        full[result.role] = result.best
        return full["subject"], full["relation"], full["object"]

    def resolve(
        self,
        queries: list[tuple[dict[str, str], dict[str, str] | None]],
        steps: int = 10,
        top_k: int = 5,
    ) -> ResolutionTrace:
        """Multi-step candidate intersection across independent queries for
        the same missing role — e.g. resolving "who chases the mouse" using
        one query, then narrowing further with "who fears water" as a
        separate, independent piece of evidence about the same entity.

        Each query is a (known, context) pair targeting the same missing
        target role. This is deliberately intersection, not score averaging:
        the entity being resolved has to be consistent with *every* piece of
        evidence simultaneously, so the candidate pool only ever shrinks, and
        one noisy step can't outvote a decisive one. `top_k` should be large
        enough that the true candidate isn't cut from an early step's list
        before later evidence gets a chance to use it — too small and a
        single step's cutoff can silently doom the whole chain; too large and
        no step ever narrows anything.

        A step's `informative` flag is set only when it actually shrinks the
        pool (or establishes it, for the first real evidence). A step that
        changes nothing contributed no information and must not be allowed to
        look like part of a chain of reasoning that resolved something — see
        the discussion in complete_detailed's docstring about forced
        collapse manufacturing false confidence.

        If intersecting a step's candidates with the running pool would empty
        it entirely, that step is treated as *conflicting* evidence rather
        than accepted at face value: the prior pool is kept, and the step is
        marked `contradictory` (surfaced on the trace) instead of silently
        concluding "no answer." This only fires once real evidence exists —
        a step that returns nothing before any evidence has been gathered
        yet is just uninformative, not contradictory.

        Every query in the list is processed, even after the pool has
        already narrowed to one candidate — a system that stops checking new
        evidence the moment it thinks it has an answer would silently miss a
        later query that actually conflicts with that answer, which is
        exactly the blind spot this method exists to avoid.
        """
        if not queries:
            raise ValueError("resolve() requires at least one query")

        role: str | None = None
        survivors: set[str] | None = None
        trace_steps: list[ResolutionStep] = []
        any_contradiction = False

        for known, context in queries:
            missing = [r for r in TARGET_ROLES if r not in known]
            if len(missing) != 1:
                raise ValueError(f"Expected exactly 2 known target slots, got {list(known.keys())}")
            if role is None:
                role = missing[0]
            elif missing[0] != role:
                raise ValueError(
                    f"All queries in a resolution chain must target the same "
                    f"role (got '{role}' then '{missing[0]}')"
                )

            result = self.complete_detailed(known, context=context, steps=steps, top_k=top_k)
            this_candidates = {name for name, _ in result.candidates}
            contradictory = False

            if not this_candidates:
                # No data for this step (e.g. relation never stored) —
                # contributes nothing either way.
                new_survivors = survivors
                informative = False
            elif survivors is None:
                # First real evidence: establishes the starting pool.
                new_survivors = this_candidates
                informative = True
            else:
                intersected = survivors & this_candidates
                if not intersected:
                    # Would wipe out everything established so far — treat as
                    # conflicting evidence, not a reason to discard prior work.
                    new_survivors = survivors
                    informative = False
                    contradictory = True
                    any_contradiction = True
                else:
                    informative = len(intersected) < len(survivors)
                    new_survivors = intersected

            trace_steps.append(ResolutionStep(
                known=dict(known),
                context=dict(context or {}),
                result=result,
                survivors=sorted(new_survivors) if new_survivors else [],
                informative=informative,
                contradictory=contradictory,
            ))
            survivors = new_survivors

        final = sorted(survivors) if survivors else []
        return ResolutionTrace(
            role=role,
            steps=trace_steps,
            final_candidates=final,
            resolved=len(final) == 1,
            contradictory=any_contradiction,
        )

    def ground_truth_ambiguity(
        self, known: dict[str, str], context: dict[str, str] | None = None
    ) -> list[tuple[str, str, str]]:
        """Stored triples that genuinely match all known slots and context — data-level collisions.

        Independent of retrieval: if this returns more than one triple, no
        encoding or pattern-completion scheme can pick a single "correct"
        answer, because the training data itself is ambiguous under this
        cue. Passing `context` narrows the match the same way it narrows a
        live query — two triples that collide on `known` alone may not
        collide once `context` also has to agree.
        """
        context = context or {}
        matches = []
        for subject, relation, object_, stored_context in self.triples:
            fields = {"subject": subject, "relation": relation, "object": object_}
            if any(fields[k] != v for k, v in known.items()):
                continue
            if any(stored_context.get(k) != v for k, v in context.items()):
                continue
            matches.append((subject, relation, object_))
        return matches

    def exact_lookup(self, subject: str, relation: str, object_: str) -> torch.Tensor | None:
        """Check if a triple is stored (for verification)."""
        vec = self.encoder.encode_triple(subject, relation, object_)
        results = self.store.lookup(vec, k=1)
        if results:
            sim = self.encoder.vsa.similarity(vec, results[0][0])
            if sim.item() > 0.9:
                return results[0][0]
        return None

    @property
    def size(self) -> int:
        return len(self.store)
