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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.vsa.hopfield import HopfieldNet
from src.vsa.primitives import VSA
from src.vsa.store import AssociativeStore


ROLE_NAMES = ("subject", "relation", "object")

# Minimum gap between the top-1 and top-2 candidate scores below which a
# completion is flagged as ambiguous rather than confidently decoded.
DEFAULT_AMBIGUITY_MARGIN = 0.1


class RelationalEncoder:
    """Encode relational triples as VSA vectors and decode fillers from them."""

    def __init__(self, vsa: VSA):
        self.vsa = vsa
        self.role_vectors: dict[str, torch.Tensor] = {}
        self.entity_vectors: dict[str, torch.Tensor] = {}
        self.relation_vectors: dict[str, torch.Tensor] = {}
        self._init_roles()

    def _init_roles(self):
        for name in ROLE_NAMES:
            self.role_vectors[name] = self.vsa.make_vector()

    def entity(self, name: str) -> torch.Tensor:
        if name not in self.entity_vectors:
            self.entity_vectors[name] = self.vsa.make_vector()
        return self.entity_vectors[name]

    def relation(self, name: str) -> torch.Tensor:
        if name not in self.relation_vectors:
            self.relation_vectors[name] = self.vsa.make_vector()
        return self.relation_vectors[name]

    def filler(self, role: str, name: str) -> torch.Tensor:
        return self.relation(name) if role == "relation" else self.entity(name)

    def candidates_for(self, role: str) -> dict[str, torch.Tensor]:
        return self.relation_vectors if role == "relation" else self.entity_vectors

    def encode_triple(
        self, subject: str, relation: str, object_: str
    ) -> torch.Tensor:
        """Encode (subject, relation, object) as a bundled VSA vector.

        encode = bind(role_subject, vec(subject))
               + bind(role_relation, vec(relation))
               + bind(role_object, vec(object))
        """
        sv = self.vsa.bind(self.role_vectors["subject"], self.entity(subject))
        rv = self.vsa.bind(self.role_vectors["relation"], self.relation(relation))
        ov = self.vsa.bind(self.role_vectors["object"], self.entity(object_))
        return self.vsa.bundle([sv, rv, ov])

    def decode_filler_topk(
        self, state: torch.Tensor, role: str, candidates: dict[str, torch.Tensor], k: int = 3
    ) -> list[tuple[str, float]]:
        """Unbind *role* from *state*, return the top-k closest candidates, best first."""
        unbound = self.vsa.unbind(state, self.role_vectors[role])
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
        """Decode all three roles from a retrieved state vector."""
        subject = self.decode_filler(state, "subject", self.entity_vectors)
        relation = self.decode_filler(state, "relation", self.relation_vectors)
        object_ = self.decode_filler(state, "object", self.entity_vectors)
        return subject, relation, object_

    def encode_query(
        self, known: dict[str, str]
    ) -> torch.Tensor:
        """Encode a partial triple from known slots. Missing slots are omitted.

        known: {"subject": "cat", "relation": "chases"}
               (object is the slot to retrieve)
        """
        components = []
        for role in ROLE_NAMES:
            if role in known:
                components.append(self.vsa.bind(self.role_vectors[role], self.filler(role, known[role])))
        if not components:
            return torch.zeros(self.vsa.dim)
        return self.vsa.bundle(components)


@dataclass
class CompletionResult:
    """Result of completing one missing slot of a triple."""

    role: str
    best: str | None
    confidence: float
    candidates: list[tuple[str, float]] = field(default_factory=list)
    ambiguous: bool = False


class RelationalMemory:
    """Relational memory partitioned by relation, with Hebbian (Hopfield) storage.

    One Hopfield net per relation stores the triples that use that relation.
    Retrieval when the relation is known queries that net directly. Retrieval
    when the relation itself is the missing slot broadcasts the query across
    every net and scores each candidate relation by how well its recalled
    state reconstructs the *known* slots (subject and/or object) — the net
    whose stored triples best explain the known slots wins.
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
        self.triples: list[tuple[str, str, str]] = []

    def _net_for(self, relation: str) -> HopfieldNet:
        net = self.nets.get(relation)
        if net is None:
            net = HopfieldNet(self.dim, retrieval_mode="modern", modern_beta=self.hopfield_beta)
            self.nets[relation] = net
        return net

    def store_triple(self, subject: str, relation: str, object_: str):
        vec = self.encoder.encode_triple(subject, relation, object_)
        self._net_for(relation).store(vec)
        self.store.insert(vec)
        self.triples.append((subject, relation, object_))

    def _infer_relation(self, known: dict[str, str], steps: int, top_k: int) -> CompletionResult:
        query = self.encoder.encode_query(known)
        scores: list[tuple[str, float]] = []
        for relation, net in self.nets.items():
            if len(net) == 0:
                continue
            recalled = net.recall(query, steps=steps)
            conf = 0.0
            for role, value in known.items():
                unbound = self.encoder.vsa.unbind(recalled, self.encoder.role_vectors[role])
                conf += self.encoder.vsa.similarity(unbound, self.encoder.filler(role, value)).item()
            scores.append((relation, conf))

        if not scores:
            return CompletionResult(role="relation", best=None, confidence=0.0)

        scores.sort(key=lambda pair: -pair[1])
        candidates = scores[:top_k]
        best, best_conf = candidates[0]
        ambiguous = len(candidates) > 1 and (candidates[0][1] - candidates[1][1]) < self.ambiguity_margin * len(known)
        return CompletionResult(role="relation", best=best, confidence=best_conf, candidates=candidates, ambiguous=ambiguous)

    def _infer_slot(self, role: str, known: dict[str, str], steps: int, top_k: int) -> CompletionResult:
        net = self.nets.get(known.get("relation"))
        if net is None or len(net) == 0:
            return CompletionResult(role=role, best=None, confidence=0.0)

        query = self.encoder.encode_query(known)
        recalled = net.recall(query, steps=steps)
        candidates = self.encoder.decode_filler_topk(recalled, role, self.encoder.candidates_for(role), k=top_k)
        best, best_score = candidates[0]
        ambiguous = len(candidates) > 1 and (candidates[0][1] - candidates[1][1]) < self.ambiguity_margin
        return CompletionResult(role=role, best=best, confidence=best_score, candidates=candidates, ambiguous=ambiguous)

    def complete_detailed(
        self, known: dict[str, str], steps: int = 10, top_k: int = 3
    ) -> CompletionResult:
        """Complete the one missing slot of a partial triple, with confidence/ambiguity info.

        `ambiguous` is set from two independent signals: whether the stored
        triples themselves genuinely contain more than one match for `known`
        (the authoritative signal — `ground_truth_ambiguity`), or-ed with a
        score-gap heuristic on the recalled candidates. The ground-truth
        check matters because a sharp (high-beta) Hopfield recall collapses
        genuine ties to whichever candidate has a slightly higher score from
        vector noise, which would otherwise hide real ambiguity.
        """
        missing = [role for role in ROLE_NAMES if role not in known]
        if len(missing) != 1:
            raise ValueError(f"Expected exactly 2 known slots, got {list(known.keys())}")
        role = missing[0]
        if role == "relation":
            result = self._infer_relation(known, steps=steps, top_k=top_k)
        else:
            result = self._infer_slot(role, known, steps=steps, top_k=top_k)
        if len(self.ground_truth_ambiguity(known)) > 1:
            result.ambiguous = True
        return result

    def complete(
        self, known: dict[str, str], steps: int = 10
    ) -> tuple[str, str, str]:
        """Complete a partial triple via Hopfield pattern completion."""
        result = self.complete_detailed(known, steps=steps)
        full = dict(known)
        full[result.role] = result.best
        return full["subject"], full["relation"], full["object"]

    def ground_truth_ambiguity(self, known: dict[str, str]) -> list[tuple[str, str, str]]:
        """Stored triples that genuinely match all known slots — data-level collisions.

        Independent of retrieval: if this returns more than one triple, no
        encoding or pattern-completion scheme can pick a single "correct"
        answer, because the training data itself is ambiguous under this cue.
        """
        return [
            triple for triple in self.triples
            if all(triple[ROLE_NAMES.index(role)] == value for role, value in known.items())
        ]

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
