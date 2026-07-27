"""Test classical Hopfield mode (iterative attractor) vs modern mode (softmax blending).

Classical mode: state = tanh(beta * W @ state), iterated until convergence.
  - Attractor dynamics pick ONE pattern, not an average.
  - Energy after convergence measures confidence.
  - W is the Hebbian outer-product sum, stored by every store() call.

Modern mode: softmax(beta * cosine(query, patterns)) @ patterns.
  - Blends all matching patterns.
  - No confidence signal (always produces a peaked distribution).

Tests:
  1. Fact recall accuracy — both modes on (relation, subject) → object
  2. Relation inference — both modes on (subject, object) → relation
  3. Chaining stability — both modes on next-token prediction
  4. Energy as confidence — does classical energy predict correctness?
"""

from __future__ import annotations

import random
import statistics
from collections import Counter

import torch

from src.vsa.primitives import VSA
from src.vsa.hopfield import HopfieldNet


# ─── helpers ────────────────────────────────────────────────────────

def make_fact_data(n_entities=50, n_relations=10, pairs_per_rel=20, seed=42):
    random.seed(seed)
    entities = [f"E{i}" for i in range(n_entities)]
    relations = [f"R{i}" for i in range(n_relations)]
    facts = {}
    for rel in relations:
        pairs = []
        used_subj = set()
        while len(pairs) < pairs_per_rel:
            s = random.choice(entities)
            o = random.choice(entities)
            if s != o and s not in used_subj:
                used_subj.add(s)
                pairs.append((s, o))
        facts[rel] = pairs
    return entities, relations, facts


def make_seq_data(text, context_width=3):
    tokens = [text[i:i+2].ljust(2, "_") for i in range(0, len(text), 2)]
    patterns = [(tokens[i:i+context_width], tokens[i+context_width])
                for i in range(len(tokens) - context_width)]
    return tokens, patterns


class FactEncoder:
    def __init__(self, vsa):
        self.vsa = vsa
        self.ir = vsa.make_vector()
        self.or_ = vsa.make_vector()
        self.cache = {}

    def tv(self, t):
        if t not in self.cache:
            self.cache[t] = self.vsa.make_vector()
        return self.cache[t]

    def encode_pair(self, s, o):
        return self.vsa.bundle([self.vsa.bind(self.ir, self.tv(s)), self.vsa.bind(self.or_, self.tv(o))])

    def encode_soq(self, s, o):
        return self.vsa.bundle([self.vsa.bind(self.ir, self.tv(s)), self.vsa.bind(self.or_, self.tv(o))])

    def encode_sq(self, s):
        return self.vsa.bind(self.ir, self.tv(s))

    def decode(self, vec, role):
        ub = self.vsa.unbind(vec, role)
        names = sorted(self.cache.keys())
        vs_ = torch.stack([self.cache[n] for n in names])
        return names[self.vsa.similarity(ub, vs_).argmax().item()]


class SeqEncoder:
    def __init__(self, vsa, context_width):
        self.vsa = vsa
        self.ctx_roles = [vsa.make_vector() for _ in range(context_width)]
        self.out_role = vsa.make_vector()
        self.cache = {}

    def tv(self, t):
        if t not in self.cache:
            self.cache[t] = self.vsa.make_vector()
        return self.cache[t]

    def encode(self, ctx, nxt):
        ctx_v = self.vsa.bundle([self.vsa.bind(self.ctx_roles[i], self.tv(ctx[i])) for i in range(len(ctx))])
        return self.vsa.bundle([ctx_v, self.vsa.bind(self.out_role, self.tv(nxt))])

    def encode_query(self, ctx):
        return self.vsa.bundle([self.vsa.bind(self.ctx_roles[i], self.tv(ctx[i])) for i in range(len(ctx))])

    def decode(self, vec):
        ub = self.vsa.unbind(vec, self.out_role)
        names = sorted(self.cache.keys())
        vs_ = torch.stack([self.cache[n] for n in names])
        return names[self.vsa.similarity(ub, vs_).argmax().item()]


# ─── test 1: fact recall (modern vs classical) ─────────────────────

def recall_fact(net, query, mode, steps=5, beta=1.0):
    """Single fact recall from a HopfieldNet."""
    if mode == "modern":
        rec = net.recall(query, steps=1, beta=50.0)  # modern ignores beta
    else:
        rec = net.recall(query, steps=steps, beta=beta)
    return rec


def test_fact_recall(entities, relations, facts, vsa, dim=1000):
    print("─── TEST 1: FACT RECALL ACCURACY (modern vs classical) ───")
    fe = FactEncoder(vsa)

    for mode, beta, steps in [("modern", 50.0, 1), ("classical", 1.0, 5), ("classical", 0.5, 10)]:
        correct = 0
        total = 0
        for rel, pairs in facts.items():
            net = HopfieldNet(dim, retrieval_mode="modern" if mode == "modern" else "classical",
                              modern_beta=50.0 if mode == "modern" else 1.0)
            for s, o in pairs:
                net.store(fe.encode_pair(s, o))

            for s, o in pairs:
                q = fe.encode_sq(s)
                rec = recall_fact(net, q, mode, steps=steps, beta=beta)
                decoded = fe.decode(rec, fe.or_)
                if decoded == o:
                    correct += 1
                total += 1

        acc = 100 * correct / total
        print(f"  {mode:15s} β={beta:<5} steps={steps:<3}  accuracy={acc:.1f}%")
    print()


# ─── test 2: relation inference ─────────────────────────────────────

def test_rel_inference(entities, relations, facts, vsa, dim=1000):
    print("─── TEST 2: RELATION INFERENCE (subject, object) → relation ───")
    fe = FactEncoder(vsa)

    # Build nets for each relation
    for mode, beta, steps, label in [("modern", 50.0, 1, "modern β=50"),
                                       ("classical", 1.0, 5, "classical β=1 s=5"),
                                       ("classical", 0.5, 10, "classical β=0.5 s=10")]:
        nets = {}
        for rel, pairs in facts.items():
            net = HopfieldNet(dim, retrieval_mode="modern" if mode == "modern" else "classical",
                              modern_beta=50.0 if mode == "modern" else 1.0)
            for s, o in pairs:
                net.store(fe.encode_pair(s, o))
            nets[rel] = net

        correct = 0
        total = 0
        for rel, pairs in facts.items():
            for s, o in pairs:
                q = fe.encode_soq(s, o)
                best_rel = None
                best_conf = -1
                for rn, net in nets.items():
                    if len(net) == 0:
                        continue
                    rec = recall_fact(net, q, mode, steps=steps, beta=beta)
                    s_d = fe.decode(rec, fe.ir)
                    o_d = fe.decode(rec, fe.or_)
                    conf = (1 if s_d == s else 0) + (1 if o_d == o else 0)
                    if conf > best_conf:
                        best_conf = conf
                        best_rel = rn
                if best_rel == rel:
                    correct += 1
                total += 1

        acc = 100 * correct / total
        print(f"  {label:25s}  accuracy={acc:.1f}%")

        # Track energy if classical
        if mode == "classical":
            en_correct, en_wrong = [], []
            for rel, pairs in facts.items():
                for s, o in pairs:
                    q = fe.encode_soq(s, o)
                    eng = nets[rel].energy(q)
                    en_correct.append(eng.item())
                    # Wrong relation
                    for rn in [r for r in relations if r != rel]:
                        eng_w = nets[rn].energy(q)
                        en_wrong.append(eng_w.item())
                        break  # just one wrong net per query
                    break  # just first pair per relation
                break  # just first relation
            if en_correct and en_wrong:
                print(f"  {'':25s}  mean energy correct={statistics.mean(en_correct):.2f}  "
                      f"wrong={statistics.mean(en_wrong):.2f}")
    print()


# ─── test 3: chaining stability ─────────────────────────────────────

def test_chaining(text, vsa, dim=1000, context_width=3, max_steps=30):
    print("─── TEST 3: CHAINING STABILITY ───")
    se = SeqEncoder(vsa, context_width)

    for mode, beta, steps, label in [("modern", 50.0, 1, "modern"),
                                       ("classical", 1.0, 5, "classical β=1 s=5"),
                                       ("classical", 0.3, 15, "classical β=0.3 s=15")]:
        tokens = [text[i:i+2].ljust(2, "_") for i in range(0, len(text), 2)]
        patterns = [(tokens[i:i+context_width], tokens[i+context_width])
                    for i in range(len(tokens) - context_width)]

        net = HopfieldNet(dim, retrieval_mode="modern" if mode == "modern" else "classical",
                          modern_beta=50.0 if mode == "modern" else 1.0)
        for ctx, nxt in patterns:
            net.store(se.encode(ctx, nxt))

        ctx = list(tokens[:context_width])
        gen = list(ctx)
        errors = 0
        energies = []
        for step in range(max_steps):
            q = se.encode_query(ctx)
            if mode == "modern":
                rec = net.recall(q, steps=1, beta=50.0)
            else:
                rec = net.recall(q, steps=steps, beta=beta)
                eng = net.energy(rec)
                energies.append(eng.item())
            decoded = se.decode(rec)
            gen.append(decoded)

            if step + context_width < len(tokens) and decoded != tokens[step + context_width]:
                errors += 1

            ctx = ctx[1:] + [decoded]

        raw = "".join(gen).replace("_", " ")
        print(f"  {label:25s}  errors={errors}/{max_steps}  "
              f"gen={raw[:50]}..."
              f"{'  mean_energy='+str(round(statistics.mean(energies),4)) if energies else ''}")
    print()


# ─── test 4: energy as confidence ───────────────────────────────────

def test_energy_confidence(entities, relations, facts, vsa, dim=1000):
    print("─── TEST 4: ENERGY AS CONFIDENCE ───")
    fe = FactEncoder(vsa)

    # Build one relation's net
    rel = relations[0]
    pairs = facts[rel]
    net = HopfieldNet(dim, retrieval_mode="classical", modern_beta=1.0)
    for s, o in pairs:
        net.store(fe.encode_pair(s, o))

    # Query with stored pairs (should have low energy → stored pattern)
    stored_energies = []
    for s, o in pairs[:10]:
        q = fe.encode_soq(s, o)
        rec = net.recall(q, steps=5, beta=1.0)
        stored_energies.append(net.energy(rec).item())
    print(f"  Energy for EXACT (s,o) matches (10 queries): mean={statistics.mean(stored_energies):.4f}  "
          f"min={min(stored_energies):.4f}  max={max(stored_energies):.4f}")

    # Query with partial matches (s only)
    partial_energies = []
    for s, o in pairs[:10]:
        q = fe.encode_sq(s)  # no object
        rec = net.recall(q, steps=5, beta=1.0)
        partial_energies.append(net.energy(rec).item())
    print(f"  Energy for PARTIAL (s only)  (10 queries): mean={statistics.mean(partial_energies):.4f}  "
          f"min={min(partial_energies):.4f}  max={max(partial_energies):.4f}")

    # Query with random vectors (should have high energy)
    random_energies = []
    for _ in range(10):
        v = vsa.make_vector()
        rec = net.recall(v, steps=5, beta=1.0)
        random_energies.append(net.energy(rec).item())
    print(f"  Energy for RANDOM queries     (10 queries): mean={statistics.mean(random_energies):.4f}  "
          f"min={min(random_energies):.4f}  max={max(random_energies):.4f}")
    print()


# ─── run ────────────────────────────────────────────────────────────

def run():
    random.seed(42)
    torch.manual_seed(42)
    vsa = VSA(dim=1000)

    entities, relations, facts = make_fact_data()

    test_fact_recall(entities, relations, facts, vsa)
    test_rel_inference(entities, relations, facts, vsa)

    text = ("the quick brown fox jumps over the lazy dog near the bank of the river "
            "while the small grey mouse runs through the green field to find some "
            "cheese and nuts for the long cold winter ahead")
    test_chaining(text, vsa, context_width=3)

    test_energy_confidence(entities, relations, facts, vsa)


if __name__ == "__main__":
    run()
