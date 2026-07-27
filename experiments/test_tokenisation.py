"""Compare atomic vs tokenised entity vectors for Hopfield relational reasoning.

Tests whether tokenisation changes interference patterns in the
VSA binding + Hopfield pattern completion pipeline.

Usage:  python -m experiments.test_tokenisation
"""

import random
import statistics
import time

import torch

from src.vsa.primitives import VSA
from src.vsa.hopfield import HopfieldNet

ENTITIES = [
    "cat", "dog", "mouse", "bird", "fish",
    "lion", "tiger", "wolf", "fox", "eagle",
    "shark", "whale", "bear", "deer", "rabbit",
    "horse", "sheep", "goat", "owl", "hawk",
    "elephant", "monkey", "turtle", "frog", "lizard",
]
RELATIONS = [
    "chases", "fears", "likes", "bigger-than", "smaller-than",
    "eats", "hunts", "hides-from", "sleeps-near", "competes-with",
]

ROLE_NAMES = ("subject", "relation", "object")


def tokenise(name: str, mode: str = "char-1gram"):
    if mode == "char-1gram":
        return list(name)
    elif mode == "char-bigram":
        return [name[i:i+2] for i in range(len(name) - 1)] or [name]
    elif mode == "atomic":
        return [name]  # whole name = one token (same as current)
    else:
        raise ValueError(f"Unknown tokenisation mode: {mode}")


class TestEncoder:
    """Minimal encoder for the test — same as RelationalEncoder but with
    configurable entity vector construction."""

    def __init__(self, vsa: VSA, mode: str = "atomic"):
        self.vsa = vsa
        self.mode = mode
        self.role_vectors = {r: vsa.make_vector() for r in ROLE_NAMES}
        self._token_cache: dict[str, torch.Tensor] = {}
        self._entity_cache: dict[str, torch.Tensor] = {}
        self._relation_cache: dict[str, torch.Tensor] = {}

    def _token_vec(self, token: str) -> torch.Tensor:
        if token not in self._token_cache:
            self._token_cache[token] = self.vsa.make_vector()
        return self._token_cache[token]

    def entity(self, name: str) -> torch.Tensor:
        if name not in self._entity_cache:
            if self.mode == "atomic":
                v = self.vsa.make_vector()
            else:
                tokens = tokenise(name, self.mode)
                vecs = [self._token_vec(t) for t in tokens]
                v = self.vsa.bundle(vecs)
            self._entity_cache[name] = v
        return self._entity_cache[name]

    def relation(self, name: str) -> torch.Tensor:
        if name not in self._relation_cache:
            self._relation_cache[name] = self.vsa.make_vector()
        return self._relation_cache[name]

    def encode_triple(self, s: str, r: str, o: str) -> torch.Tensor:
        sv = self.vsa.bind(self.role_vectors["subject"], self.entity(s))
        rv = self.vsa.bind(self.role_vectors["relation"], self.relation(r))
        ov = self.vsa.bind(self.role_vectors["object"], self.entity(o))
        return self.vsa.bundle([sv, rv, ov])

    def encode_query(self, known: dict) -> torch.Tensor:
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

    def decode_filler(self, state, role, candidates):
        unbound = self.vsa.unbind(state, self.role_vectors[role])
        names = list(candidates.keys())
        vecs = torch.stack([candidates[n] for n in names])
        sims = self.vsa.similarity(unbound, vecs)
        return names[sims.argmax().item()]

    def decode_triple(self, state):
        s = self.decode_filler(state, "subject", self._entity_cache)
        r = self.decode_filler(state, "relation", self._relation_cache)
        o = self.decode_filler(state, "object", self._entity_cache)
        return s, r, o


def generate_triples(entities, relations, n, seed=42):
    random.seed(seed)
    generated = set()
    triples = []
    while len(triples) < n:
        s = random.choice(entities)
        r = random.choice(relations)
        o = random.choice(entities)
        if s != o and (s, r, o) not in generated:
            generated.add((s, r, o))
            triples.append((s, r, o))
    return triples


def test_mode(vsa, mode, triples, relations, dim=10000, use_polygraph=False):
    """Run completion test. Returns (subj%, rel%, obj%).

    When *use_polygraph* is True, one HopfieldNet per relation is created
    instead of a single shared net. Queries with a known relation dispatch
    directly to the correct sub-net; queries with unknown relation broadcast
    to all sub-nets and pick the best match by confidence scoring.
    """
    encoder = TestEncoder(vsa, mode=mode)

    if use_polygraph:
        nets = {r: HopfieldNet(dim, retrieval_mode="modern", modern_beta=50.0)
                for r in relations}
        for s, r, o in triples:
            vec = encoder.encode_triple(s, r, o)
            nets[r].store(vec)
    else:
        hopfield = HopfieldNet(dim, retrieval_mode="modern", modern_beta=50.0)
        for s, r, o in triples:
            vec = encoder.encode_triple(s, r, o)
            hopfield.store(vec)

    correct_s = correct_r = correct_o = 0
    for s, r, o in triples:
        # subject from (relation, object) — relation known
        if use_polygraph:
            net = nets[r]
        q = encoder.encode_query({"relation": r, "object": o})
        recalled = (net if use_polygraph else hopfield).recall(q, beta=50.0)
        s1, _, _ = encoder.decode_triple(recalled)
        if s1 == s:
            correct_s += 1

        # relation from (subject, object) — relation UNKNOWN, broadcast
        q = encoder.encode_query({"subject": s, "object": o})
        if use_polygraph:
            # broadcast to all sub-nets, pick best by decoded-entity match
            best_r = None
            best_conf = -1
            for rn, net in nets.items():
                if len(net) == 0:
                    continue
                rec = net.recall(q, beta=50.0)
                s_dec, rn_dec, o_dec = encoder.decode_triple(rec)
                # confidence: fraction of known slots (s, o) that match
                match = (1 if s_dec == s else 0) + (1 if o_dec == o else 0)
                if match > best_conf:
                    best_conf = match
                    best_r = rn_dec
            r1 = best_r if best_r is not None else "__none__"
        else:
            recalled = hopfield.recall(q, beta=50.0)
            _, r1, _ = encoder.decode_triple(recalled)
        if r1 == r:
            correct_r += 1

        # object from (subject, relation) — relation known
        if use_polygraph:
            net = nets[r]
        q = encoder.encode_query({"subject": s, "relation": r})
        recalled = (net if use_polygraph else hopfield).recall(q, beta=50.0)
        _, _, o1 = encoder.decode_triple(recalled)
        if o1 == o:
            correct_o += 1

    n = len(triples)
    return (100 * correct_s / n, 100 * correct_r / n, 100 * correct_o / n)


def test_entity_similarity(mode, entities=ENTITIES):
    """Measure average cosine similarity between pairs of entity vectors
    in the given mode. Higher similarity = more interference potential."""
    vsa = VSA(dim=10000)
    encoder = TestEncoder(vsa, mode=mode)
    sims = []
    for i, e1 in enumerate(entities):
        v1 = encoder.entity(e1)
        for e2 in entities[i+1:]:
            v2 = encoder.entity(e2)
            sim = vsa.similarity(v1, v2).item()
            sims.append(sim)
    return statistics.mean(sims), statistics.median(sims), max(sims)


def run():
    print("Entity similarity across modes:")
    print(f"{'mode':15s} {'mean_cos':>8s} {'median_cos':>10s} {'max_cos':>7s}")
    for mode in ("atomic", "char-1gram", "char-bigram"):
        mean_sim, med_sim, max_sim = test_entity_similarity(mode)
        print(f"{mode:15s} {mean_sim:>8.4f} {med_sim:>10.4f} {max_sim:>7.4f}")
    print()

    # Generate triples once, reuse across modes
    triples = generate_triples(ENTITIES, RELATIONS, 200, seed=42)

    print(f"{'N':>5s} {'mode':20s} {'subject':>7s} {'relation':>8s} {'object':>7s}")
    print("-" * 53)

    for n in [10, 50, 100, 200]:
        t = triples[:n]
        for mode in ("atomic", "char-bigram"):
            # single net baseline
            vsa = VSA(dim=10000)
            subj, rel, obj = test_mode(vsa, mode, t, RELATIONS, use_polygraph=False)
            print(f"{n:5d} {'single-' + mode:20s} {subj:>6.1f}% {rel:>7.1f}% {obj:>6.1f}%")

            # polygraph
            vsa = VSA(dim=10000)
            subj, rel, obj = test_mode(vsa, mode, t, RELATIONS, use_polygraph=True)
            print(f"{n:5d} {'polygraph-' + mode:20s} {subj:>6.1f}% {rel:>7.1f}% {obj:>6.1f}%")

    # Also print the number of unique tokens per mode
    print()
    for mode in ("char-1gram", "char-bigram"):
        tokens = set()
        for e in ENTITIES:
            for t in tokenise(e, mode):
                tokens.add(t)
        print(f"{mode}: {len(tokens)} unique tokens across {len(ENTITIES)} entities")


if __name__ == "__main__":
    run()
