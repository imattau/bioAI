"""Compare five VSA vector models on fact recall and relation inference.

All models share the same encoder structure, storage, and retrieval
(softmax attention over stored patterns). Only the vector primitives
(make_vector, bind, bundle, similarity) change.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter

import torch

from src.vsa.primitives import VSA as BSC_VSA


# ─── data ───────────────────────────────────────────────────────────

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


# ─── vector models ───────────────────────────────────────────────────

class ModelBase:
    name = "base"
    def make_vector(self, dim): raise NotImplementedError
    def bind(self, a, b): raise NotImplementedError
    def bundle(self, items): raise NotImplementedError
    def similarity(self, a, b): raise NotImplementedError
    def bits_per_vector(self, dim): raise NotImplementedError

    def encode_pair(self, ir, or_, s, o):
        return self.bundle([self.bind(ir, s), self.bind(or_, o)])

    def encode_soq(self, ir, or_, s, o):
        return self.bundle([self.bind(ir, s), self.bind(or_, o)])

    def encode_sq(self, ir, s):
        return self.bind(ir, s)

    def decode(self, vec, role, cache):
        ub = self.bind(vec, role)  # self-inverse for bipolar; adjust for other models
        names = sorted(cache.keys())
        vs_ = torch.stack([cache[n] for n in names])
        sims = torch.tensor([self.similarity(ub, v).item() for v in vs_])
        return names[sims.argmax().item()]

    def unbind(self, a, b):
        """Default: bind is self-inverse for bipolar. Override if needed."""
        return self.bind(a, b)


# 1. BSC Bipolar (baseline) — ±1 dense
class ModelBSC(ModelBase):
    name = "BSC (baseline)"
    def make_vector(self, dim):
        return torch.where(torch.rand(dim) >= 0.5, 1.0, -1.0)
    def bind(self, a, b):
        return a * b
    def bundle(self, items):
        return torch.sign(torch.stack(items).sum(dim=0))
    def similarity(self, a, b):
        return (a * b).sum() / a.numel()
    def bits_per_vector(self, dim):
        return dim


# 2. SBDR Sparse Binary — {0,1}, 3% active
class ModelSparseBin(ModelBase):
    name = "SBDR sparse 3%"
    def __init__(self, sparsity=0.03):
        self.sparsity = sparsity
        self._perm_cache = {}
    def _perm(self, role, dim):
        key = id(role)
        if key not in self._perm_cache:
            # Random permutation = torch.roll with random shift
            self._perm_cache[key] = int(torch.rand(1).item() * dim)
        return self._perm_cache[key]
    def make_vector(self, dim):
        v = torch.zeros(dim)
        n = max(1, int(dim * self.sparsity))
        idx = torch.randperm(dim)[:n]
        v[idx] = 1.0
        return v
    def bind(self, a, b):
        # Permutation-based binding for sparse: roll by hash(b)
        shift = int(torch.tensor(b).sum().item()) % a.numel() if isinstance(b, torch.Tensor) else 0
        return torch.roll(a, shifts=shift)
    def bundle(self, items):
        s = torch.stack(items).sum(dim=0)
        return (s >= 1).float()
    def similarity(self, a, b):
        inter = ((a > 0) & (b > 0)).sum().float()
        na = (a > 0).sum().float().clamp(min=1)
        nb = (b > 0).sum().float().clamp(min=1)
        return inter / min(na, nb)
    def bits_per_vector(self, dim):
        return max(1, int(dim * self.sparsity * math.log2(dim) / 8))


# 3. SBDR Sparse Bipolar — ±1 when active, 0 otherwise (3%)
class ModelSparseBip(ModelBase):
    name = "SBDR sparse bip 3%"
    def __init__(self, sparsity=0.03):
        self.sparsity = sparsity
    def make_vector(self, dim):
        v = torch.zeros(dim)
        n = max(1, int(dim * self.sparsity))
        idx = torch.randperm(dim)[:n]
        v[idx] = torch.where(torch.rand(n) >= 0.5, 1.0, -1.0)
        return v
    def bind(self, a, b):
        return a * b
    def bundle(self, items):
        s = torch.stack(items).sum(dim=0)
        return torch.where(s > 0, 1.0, torch.where(s < 0, -1.0, torch.zeros_like(s)))
    def similarity(self, a, b):
        a_mask = a != 0
        b_mask = b != 0
        inter = (a_mask & b_mask)
        if inter.sum() == 0:
            return 0.0
        agreed = ((a * b) > 0) & inter
        return agreed.sum().float() / max(1, int(inter.sum()))
    def bits_per_vector(self, dim):
        return max(1, int(dim * self.sparsity * 2))  # sign + position


# 4. HRR — real-valued, circular convolution as bind
class ModelHRR(ModelBase):
    name = "HRR"
    def make_vector(self, dim):
        v = torch.randn(dim)
        return v / v.norm()
    def bind(self, a, b):
        # circular convolution via FFT
        A, B = torch.fft.fft(a), torch.fft.fft(b)
        return torch.fft.ifft(A * B).real
    def unbind(self, a, b):
        A, B = torch.fft.fft(a), torch.fft.fft(b)
        return torch.fft.ifft(A * torch.conj(B)).real

    def decode(self, vec, role, cache):
        # Use UNbind (circular correlation) not bind for HRR
        ub = self.unbind(vec, role)
        names = sorted(cache.keys())
        vs_ = torch.stack([cache[n] for n in names])
        sims = torch.tensor([self.similarity(ub, v).item() for v in vs_])
        return names[sims.argmax().item()]
    def bundle(self, items):
        s = torch.stack(items).sum(dim=0)
        return s / s.norm().clamp(min=1e-8)
    def similarity(self, a, b):
        return (a * b).sum() / (a.norm().clamp(min=1e-8) * b.norm().clamp(min=1e-8))
    def bits_per_vector(self, dim):
        return dim * 32  # float32


# 5. Real-valued VSA (from v2 arch) — element-wise multiply, L2 norm
class ModelRealVSA(ModelBase):
    name = "real-valued VSA"
    def make_vector(self, dim):
        v = torch.randn(dim)
        return v / v.norm()
    def bind(self, a, b):
        return a * b
    def bundle(self, items):
        s = torch.stack(items).sum(dim=0)
        return s / s.norm().clamp(min=1e-8)
    def similarity(self, a, b):
        return (a * b).sum() / (a.norm().clamp(min=1e-8) * b.norm().clamp(min=1e-8))
    def bits_per_vector(self, dim):
        return dim * 32


# ─── test runner ────────────────────────────────────────────────────

def test_model(model, facts, dim=1000):
    """Run all measurements for one model. Returns dict of metrics."""
    # Build token/entity cache
    all_entities = list({e for pairs in facts.values() for s, o in pairs for e in (s, o)})
    cache = {e: model.make_vector(dim) for e in all_entities}

    # Role vectors
    ir = model.make_vector(dim)
    or_ = model.make_vector(dim)

    # ── Fact recall ──
    fact_correct = 0
    fact_total = 0
    for rel, pairs in facts.items():
        patterns = []
        for s, o in pairs:
            patterns.append(model.encode_pair(ir, or_, cache[s], cache[o]))

        for (s, o), pattern in zip(pairs, patterns):
            q = model.encode_sq(ir, cache[s])
            sims = torch.tensor([model.similarity(q, p).item() for p in patterns])
            weights = torch.softmax(50.0 * sims, dim=0)
            recalled = sum(w * p for w, p in zip(weights, patterns))
            decoded = model.decode(recalled, or_, cache)
            if decoded == o:
                fact_correct += 1
            fact_total += 1

    fact_acc = 100 * fact_correct / fact_total if fact_total else 0

    # ── Relation inference ──
    nets = {}
    for rel, pairs in facts.items():
        nets[rel] = [model.encode_pair(ir, or_, cache[s], cache[o]) for s, o in pairs]

    inf_correct = 0
    inf_total = 0
    for rel, pairs in facts.items():
        for s, o in pairs:
            q = model.encode_soq(ir, or_, cache[s], cache[o])
            best_rel = None
            best_conf = -1
            for rn, net_patterns in nets.items():
                sims = torch.tensor([model.similarity(q, p).item() for p in net_patterns])
                weights = torch.softmax(50.0 * sims, dim=0)
                recalled = sum(w * p for w, p in zip(weights, net_patterns))
                s_d = model.decode(recalled, ir, cache)
                o_d = model.decode(recalled, or_, cache)
                conf = (1 if s_d == s else 0) + (1 if o_d == o else 0)
                if conf > best_conf:
                    best_conf = conf
                    best_rel = rn
            if best_rel == rel:
                inf_correct += 1
            inf_total += 1

    inf_acc = 100 * inf_correct / inf_total if inf_total else 0

    # ── Collision count ──
    # For each (s, o) query across all nets, count how many nets have
    # a pattern that matches both slots (exact collision) or one slot (partial)
    cross_collisions = 0
    all_s = set(e for pairs in facts.values() for s, _ in pairs for e in (s,))
    for s in all_s:
        seen_os = {}
        for rel, pairs in facts.items():
            for s2, o2 in pairs:
                if s2 == s:
                    seen_os.setdefault(o2, []).append(rel)
        for o, rels in seen_os.items():
            if len(rels) > 1:
                cross_collisions += 1

    # ── Capacity ──
    # Store N patterns from one relation, measure accuracy vs N
    capacity_n = 0
    first_rel_pairs = list(facts.values())[0]
    for n_test in [10, 20, 50, 100, 200]:
        test_p = first_rel_pairs[:n_test]
        test_patterns = [model.encode_pair(ir, or_, cache[s], cache[o]) for s, o in test_p]
        correct = 0
        for (s, o), p in zip(test_p, test_patterns):
            q = model.encode_sq(ir, cache[s])
            sims = torch.tensor([model.similarity(q, tp).item() for tp in test_patterns])
            weights = torch.softmax(50.0 * sims, dim=0)
            recalled = sum(w * tp for w, tp in zip(weights, test_patterns))
            decoded = model.decode(recalled, or_, cache)
            if decoded == o:
                correct += 1
        acc = 100 * correct / len(test_p)
        if acc >= 95:
            capacity_n = n_test

    # ── Storage estimate ──
    n_patterns = sum(len(p) for p in nets.values())
    storage = n_patterns * model.bits_per_vector(dim) / 8  # bytes
    compressed = model.bits_per_vector(dim) < 16

    return {
        "name": model.name,
        "fact_accuracy": round(fact_acc, 1),
        "inf_accuracy": round(inf_acc, 1),
        "cross_collisions": cross_collisions,
        "capacity_95pct": capacity_n,
        "storage_bytes_per_pattern": round(model.bits_per_vector(dim) / 8, 1),
        "compressed": compressed,
    }


# ─── run ────────────────────────────────────────────────────────────

def run():
    random.seed(42)
    torch.manual_seed(42)

    entities, relations, facts = make_fact_data(n_entities=50, n_relations=10, pairs_per_rel=20)
    print(f"Data: {sum(len(p) for p in facts.values())} fact pairs across {len(facts)} relations")

    models = [
        ModelBSC(),
        ModelHRR(),
        ModelRealVSA(),
    ]

    results = []
    for model in models:
        print(f"  Testing {model.name}...", end=" ", flush=True)
        r = test_model(model, facts, dim=1000)
        results.append(r)
        print(f"fact={r['fact_accuracy']}%  inf={r['inf_accuracy']}%  cap={r['capacity_95pct']}")

    # Table
    print("\n" + "=" * 80)
    print(f"{'model':25s} {'fact':>6s} {'inf':>6s} {'cap@95':>7s} {'bytes/pat':>9s}")
    print("-" * 80)
    for r in results:
        cap = r["capacity_95pct"]
        cap_s = f"≥{cap}" if cap >= 200 else str(cap)
        print(f"{r['name']:25s} {r['fact_accuracy']:>5.1f}% {r['inf_accuracy']:>5.1f}% "
              f"{cap_s:>7s} {r['storage_bytes_per_pattern']:>8.1f}")
    print("=" * 80)


if __name__ == "__main__":
    run()
