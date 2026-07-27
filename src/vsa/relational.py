"""VSA relational encoding and pattern completion via Hopfield net.

Core architecture prototype (ARCHITECTURE.md Section 6):
  encode triple → Hopfield store (Hebbian) → partial query → Hopfield recall → decode filler

No backprop, no attention, no language. Pure VSA binding + pattern completion.
"""

from __future__ import annotations

import torch

from src.vsa.hopfield import HopfieldNet
from src.vsa.primitives import VSA
from src.vsa.store import AssociativeStore


ROLE_NAMES = ("subject", "relation", "object")


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

    def decode_filler(
        self, state: torch.Tensor, role: str, candidates: dict[str, torch.Tensor]
    ) -> str:
        """Unbind *role* from *state*, return closest candidate name."""
        unbound = self.vsa.unbind(state, self.role_vectors[role])
        names = list(candidates.keys())
        vectors = torch.stack([candidates[n] for n in names])
        sims = self.vsa.similarity(unbound, vectors)
        return names[sims.argmax().item()]

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
                if role == "relation":
                    vec = self.relation(known[role])
                else:
                    vec = self.entity(known[role])
                components.append(self.vsa.bind(self.role_vectors[role], vec))
        if not components:
            return torch.zeros(self.vsa.dim)
        return self.vsa.bundle(components)


class RelationalMemory:
    """Hopfield-based relational memory with Hebbian storage.

    Stores triple vectors in a Hopfield net via local outer-product updates.
    Retrieval is pattern completion via energy minimization — no gradient descent.
    """

    def __init__(self, encoder: RelationalEncoder, dim: int = 10000, hopfield_beta: float = 50.0):
        self.encoder = encoder
        self.hopfield = HopfieldNet(dim, retrieval_mode="modern", modern_beta=hopfield_beta)
        self.store = AssociativeStore(dim)
        self.hopfield_beta = hopfield_beta

    def store_triple(self, subject: str, relation: str, object_: str):
        vec = self.encoder.encode_triple(subject, relation, object_)
        self.hopfield.store(vec)
        self.store.insert(vec)

    def complete(
        self, known: dict[str, str], steps: int = 10
    ) -> tuple[str, str, str]:
        """Complete a partial triple via Hopfield pattern completion."""
        query = self.encoder.encode_query(known)
        recalled = self.hopfield.recall(query, steps=steps, beta=self.hopfield_beta)
        return self.encoder.decode_triple(recalled)

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
        return len(self.hopfield)
