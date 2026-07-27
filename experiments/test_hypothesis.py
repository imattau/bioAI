"""Test the unified hypothesis: VSA encode → Hopfield recall → decode
handles both fact retrieval and next-token prediction equivalently.

Architecture: MultiPolygraphMemory with one net per relation for facts,
plus one net for sequences. Both use the same encoding mechanism.

Five tests:
  1. Fact recall: (relation, subject) → object
  2. Relation inference: (subject, object) → relation  (polygraph broadcast)
  3. Sequence prediction: (ctx₁..ctx_N) → next token
  4. Chaining: seed → generate multiple steps
  5. Cross-task comparison: accuracy vs collision count for both tasks
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

import torch

from src.vsa.primitives import VSA
from src.vsa.hopfield import HopfieldNet

# ─── constants ──────────────────────────────────────────────────────

FACT_RELATIONS = [
    "chases", "fears", "bigger-than", "eats", "hunts",
    "hides-from", "sleeps-near", "competes-with", "likes", "smaller-than",
]

FACT_ENTITIES = [
    "cat", "dog", "mouse", "lion", "tiger", "wolf", "fox", "deer",
    "rabbit", "elephant", "bear", "shark", "whale", "eagle", "hawk",
    "snake", "frog", "owl", "goat", "horse",
]

SEQUENCE_TEXTS = {
    "short": "the cat sat on the mat and the dog ran in the park",
    "medium": (
        "the quick brown fox jumps over the lazy dog near the bank of the river "
        "while the small grey mouse runs through the green field to find some "
        "cheese and nuts for the long cold winter ahead"
    ),
    "longer": (
        "in the beginning there was the word and the word was with god and the "
        "word was god the same was in the beginning with god all things were made "
        "by him and without him was not any thing made that was made"
    ),
}


# ─── helpers ────────────────────────────────────────────────────────

def tokenise(text: str, width: int = 2) -> list[str]:
    return [text[i:i+width].ljust(width, "_") for i in range(0, len(text), width)]


def build_fact_pairs(
    entities: list[str], relations: list[str],
    pairs_per_rel: int = 20, seed: int = 42,
) -> dict[str, list[tuple[str, str]]]:
    """Generate unique (subject, object) pairs per relation."""
    random.seed(seed)
    facts: dict[str, list[tuple[str, str]]] = {}
    for rel in relations:
        pairs = []
        used_subj = set()
        attempts = 0
        while len(pairs) < pairs_per_rel and attempts < pairs_per_rel * 20:
            s = random.choice(entities)
            o = random.choice(entities)
            if s != o and s not in used_subj:
                used_subj.add(s)
                pairs.append((s, o))
            attempts += 1
        facts[rel] = pairs
    return facts


def build_n_gram_pairs(
    tokens: list[str], n: int,
) -> list[tuple[list[str], str]]:
    """Build (context_N, next_token) patterns from a token sequence."""
    patterns = []
    for i in range(len(tokens) - n):
        ctx = tokens[i:i + n]
        nxt = tokens[i + n]
        patterns.append((ctx, nxt))
    return patterns


# ─── encoder ────────────────────────────────────────────────────────

class UnifiedEncoder:
    """VSA encoder for both fact pairs and sequence n-grams."""

    def __init__(self, vsa: VSA):
        self.vsa = vsa
        # Role vectors for fact slots
        self.subj_role = vsa.make_vector()
        self.obj_role = vsa.make_vector()
        # Role vectors for sequence positions (up to 6)
        self.pos_roles = [vsa.make_vector() for _ in range(6)]
        self.seq_out_role = vsa.make_vector()
        # Token/entity vectors (shared)
        self._token_cache: dict[str, torch.Tensor] = {}
        self._tokens: set[str] = set()

    def token_vec(self, name: str) -> torch.Tensor:
        if name not in self._token_cache:
            self._token_cache[name] = self.vsa.make_vector()
            self._tokens.add(name)
        return self._token_cache[name]

    def encode_fact(self, subj: str, obj: str) -> torch.Tensor:
        return self.vsa.bundle([
            self.vsa.bind(self.subj_role, self.token_vec(subj)),
            self.vsa.bind(self.obj_role, self.token_vec(obj)),
        ])

    def encode_fact_query(self, subj: str) -> torch.Tensor:
        return self.vsa.bind(self.subj_role, self.token_vec(subj))

    def encode_fact_query_so(self, subj: str, obj: str) -> torch.Tensor:
        return self.vsa.bundle([
            self.vsa.bind(self.subj_role, self.token_vec(subj)),
            self.vsa.bind(self.obj_role, self.token_vec(obj)),
        ])

    def encode_ngram(self, ctx: list[str], nxt: str) -> torch.Tensor:
        ctx_vec = self.vsa.bundle([
            self.vsa.bind(self.pos_roles[i], self.token_vec(ctx[i]))
            for i in range(len(ctx))
        ])
        return self.vsa.bundle([
            ctx_vec,
            self.vsa.bind(self.seq_out_role, self.token_vec(nxt)),
        ])

    def encode_ngram_query(self, ctx: list[str]) -> torch.Tensor:
        return self.vsa.bundle([
            self.vsa.bind(self.pos_roles[i], self.token_vec(ctx[i]))
            for i in range(len(ctx))
        ])

    def decode_token(self, state: torch.Tensor, role: torch.Tensor) -> str:
        unbound = self.vsa.unbind(state, role)
        names = sorted(self._tokens)
        vecs = torch.stack([self._token_cache[n] for n in names])
        return names[self.vsa.similarity(unbound, vecs).argmax().item()]

    def all_tokens(self) -> list[str]:
        return sorted(self._tokens)

    def all_token_vecs(self) -> torch.Tensor:
        return torch.stack([self._token_cache[n] for n in self.all_tokens()])


# ─── tests ──────────────────────────────────────────────────────────

def test_fact_recall(facts: dict, enc: UnifiedEncoder, dim: int = 1000, beta: float = 50.0):
    """Test 1: (relation, subject) → object for each stored fact."""
    results = {}
    for rel, pairs in facts.items():
        hop = HopfieldNet(dim, retrieval_mode="modern", modern_beta=beta)
        for s, o in pairs:
            hop.store(enc.encode_fact(s, o))

        correct = 0
        for s, o in pairs:
            q = enc.encode_fact_query(s)
            recalled = hop.recall(q)
            decoded = enc.decode_token(recalled, enc.obj_role)
            if decoded == o:
                correct += 1
        results[rel] = {"pairs": len(pairs), "accuracy": round(100 * correct / len(pairs), 1)}

    overall = round(sum(r["accuracy"] for r in results.values()) / len(results), 1)
    return {"per_relation": results, "overall": overall}


def test_relation_inference(facts: dict, enc: UnifiedEncoder, dim: int = 1000, beta: float = 50.0):
    """Test 2: (subject, object) → relation via polygraph broadcast."""
    # Build one Hopfield net per relation
    nets = {}
    for rel, pairs in facts.items():
        hop = HopfieldNet(dim, retrieval_mode="modern", modern_beta=beta)
        for s, o in pairs:
            hop.store(enc.encode_fact(s, o))
        nets[rel] = hop

    all_relations = list(facts.keys())
    all_objs = enc.all_tokens()

    correct = 0
    total = 0
    for rel, pairs in facts.items():
        for s, o in pairs:
            q = enc.encode_fact_query_so(s, o)
            best_rel = None
            best_conf = -1
            for rn, net in nets.items():
                if len(net) == 0:
                    continue
                recalled = net.recall(q)
                s_dec = enc.decode_token(recalled, enc.subj_role)
                o_dec = enc.decode_token(recalled, enc.obj_role)
                conf = (1 if s_dec == s else 0) + (1 if o_dec == o else 0)
                if conf > best_conf:
                    best_conf = conf
                    best_rel = rn
            if best_rel == rel:
                correct += 1
            total += 1

    return {"accuracy": round(100 * correct / total, 1), "total": total}


def test_sequence_prediction(text: str, enc: UnifiedEncoder, dim: int = 1000,
                              beta: float = 50.0, n_values=None):
    """Test 3: (ctx₁..ctx_N) → next token at multiple N."""
    if n_values is None:
        n_values = [1, 2, 3, 4]
    tokens = tokenise(text)
    results = {}
    for n in n_values:
        if n > len(tokens) - 1:
            results[n] = {"accuracy": 0, "patterns": 0}
            continue
        patterns = build_n_gram_pairs(tokens, n)
        hop = HopfieldNet(dim, retrieval_mode="modern", modern_beta=beta)
        for ctx, nxt in patterns:
            hop.store(enc.encode_ngram(ctx, nxt))

        correct = 0
        for ctx, nxt in patterns:
            q = enc.encode_ngram_query(ctx)
            recalled = hop.recall(q)
            decoded = enc.decode_token(recalled, enc.seq_out_role)
            if decoded == nxt:
                correct += 1

        results[n] = {
            "accuracy": round(100 * correct / len(patterns), 1),
            "patterns": len(patterns),
        }
    return results


def test_chaining(text: str, enc: UnifiedEncoder, dim: int = 1000,
                   beta: float = 50.0, n_values=None, max_steps: int = 30):
    """Test 4: chaining at multiple context widths."""
    if n_values is None:
        n_values = [1, 2, 3]
    tokens = tokenise(text)
    results = {}
    for n in n_values:
        if n > len(tokens) - 1:
            continue
        patterns = build_n_gram_pairs(tokens, n)
        hop = HopfieldNet(dim, retrieval_mode="modern", modern_beta=beta)
        for ctx, nxt in patterns:
            hop.store(enc.encode_ngram(ctx, nxt))

        # Seed from corpus start
        ctx = list(tokens[:n])
        generated = list(ctx)
        first_error = max_steps

        for step in range(max_steps):
            q = enc.encode_ngram_query(ctx)
            recalled = hop.recall(q)
            decoded = enc.decode_token(recalled, enc.seq_out_role)
            generated.append(decoded)

            # Check if this step matches the original
            if step + n < len(tokens) and decoded == tokens[step + n]:
                pass  # correct
            elif step < first_error:
                first_error = step

            ctx = ctx[1:] + [decoded]

        # Detect loop end
        raw = "".join(generated).replace("_", " ")
        results[n] = {
            "first_error_step": first_error if first_error < max_steps else "none",
            "generated": raw[:60],
        }
    return results


def test_cross_task_comparison(facts: dict, text: str, enc: UnifiedEncoder,
                                dim: int = 1000, beta: float = 50.0):
    """Compare collision distributions for facts vs sequences."""
    from collections import Counter

    fact_collision_dist = {}
    for rel, pairs in facts.items():
        subj_counts = Counter(s for s, _ in pairs)
        for s, o in pairs:
            c = subj_counts[s]
            fact_collision_dist[c] = fact_collision_dist.get(c, 0) + 1

    tokens = tokenise(text)
    seq_collision_dist = {}
    for n in [1, 2, 3]:
        patterns = build_n_gram_pairs(tokens, n)
        ctx_counts = Counter(tuple(ctx) for ctx, _ in patterns)
        for ctx, nxt in patterns:
            c = ctx_counts[tuple(ctx)]
            key = f"N={n}"
            seq_collision_dist[key] = seq_collision_dist.get(key, {})
            seq_collision_dist[key][c] = seq_collision_dist[key].get(c, 0) + 1

    return {
        "fact_collision_dist": dict(sorted(fact_collision_dist.items())),
        "seq_collision_dist": {k: dict(sorted(v.items())) for k, v in seq_collision_dist.items()},
    }


# ─── main ───────────────────────────────────────────────────────────

def run():
    dim = 1000
    beta = 50.0

    print("=" * 60)
    print("HYPOTHESIS TEST: Unified VSA encode → Hopfield recall → decode")
    print("=" * 60)

    # Build data
    facts = build_fact_pairs(FACT_ENTITIES, FACT_RELATIONS, pairs_per_rel=20)
    vsa = VSA(dim=dim)
    enc = UnifiedEncoder(vsa)

    # ── Test 1: Fact recall ──
    print("\n─── TEST 1: FACT RECALL (relation, subject) → object ───")
    r1 = test_fact_recall(facts, enc, dim=dim, beta=beta)
    for rel, res in sorted(r1["per_relation"].items()):
        bar = "#" * int(res["accuracy"] / 5)
        print(f"  {rel:15s} {res['accuracy']:>5.1f}%  {bar}")
    print(f"\n  Overall fact recall: {r1['overall']:.1f}%")

    # ── Test 2: Relation inference ──
    print(f"\n─── TEST 2: RELATION INFERENCE (subject, object) → relation ───")
    r2 = test_relation_inference(facts, enc, dim=dim, beta=beta)
    print(f"  Accuracy: {r2['accuracy']}%  (over {r2['total']} queries)")

    # ── Test 3: Sequence prediction ──
    print(f"\n─── TEST 3: SEQUENCE PREDICTION (ctx₁..ctx_N) → next token ───")
    for name, text in SEQUENCE_TEXTS.items():
        r3 = test_sequence_prediction(text, enc, dim=dim, beta=beta, n_values=[1, 2, 3, 4])
        parts = [f"  N={n}: {res['accuracy']}%" for n, res in sorted(r3.items())]
        print(f"  {name:10s}: " + "  |  ".join(parts))

    # ── Test 4: Chaining ──
    print(f"\n─── TEST 4: CHAINING ───")
    for name, text in SEQUENCE_TEXTS.items():
        r4 = test_chaining(text, enc, dim=dim, beta=beta, n_values=[1, 2, 3])
        for n, res in sorted(r4.items()):
            print(f"  {name:10s} N={n}: err@{res['first_error_step']}  gen={res['generated']}")

    # ── Test 5: Cross-task comparison ──
    print(f"\n─── TEST 5: CROSS-TASK COMPARISON ───")
    r5 = test_cross_task_comparison(facts, SEQUENCE_TEXTS["medium"], enc, dim=dim, beta=beta)
    print(f"  Fact collision distribution (complete):")
    for c, count in sorted(r5["fact_collision_dist"].items()):
        print(f"    {c} completion(s) per query: {count} queries")
    print(f"  Sequence collision distribution (n-gram):")
    for key, subdict in sorted(r5["seq_collision_dist"].items()):
        for c, count in sorted(subdict.items()):
            print(f"    {key}, {c} completion(s) per query: {count} queries")

    # ── Test 5b: Direct accuracy-vs-collision comparison ──
    print(f"\n─── TEST 5b: ACCURACY vs COLLISION COUNT ───")
    # Measure accuracy per collision count for facts
    fact_acc_by_collision = {}
    for rel, pairs in facts.items():
        hop = HopfieldNet(dim, retrieval_mode="modern", modern_beta=beta)
        for s, o in pairs:
            hop.store(enc.encode_fact(s, o))

        subj_counts = Counter(s for s, _ in pairs)
        for s, o in pairs:
            q = enc.encode_fact_query(s)
            recalled = hop.recall(q)
            decoded = enc.decode_token(recalled, enc.obj_role)
            c = subj_counts[s]
            if c not in fact_acc_by_collision:
                fact_acc_by_collision[c] = {"correct": 0, "total": 0}
            fact_acc_by_collision[c]["total"] += 1
            if decoded == o:
                fact_acc_by_collision[c]["correct"] += 1

    # Measure accuracy per collision count for sequences (N=1 context)
    tokens = tokenise(SEQUENCE_TEXTS["medium"])
    patterns = build_n_gram_pairs(tokens, 1)
    hop = HopfieldNet(dim, retrieval_mode="modern", modern_beta=beta)
    for ctx, nxt in patterns:
        hop.store(enc.encode_ngram(ctx, nxt))
    ctx_tuples = [tuple(c) for c, _ in patterns]
    ctx_counts = Counter(ctx_tuples)
    seq_acc_by_collision = {}
    for (ctx, nxt), ctx_t in zip(patterns, ctx_tuples):
        q = enc.encode_ngram_query(ctx)
        recalled = hop.recall(q)
        decoded = enc.decode_token(recalled, enc.seq_out_role)
        c = ctx_counts[ctx_t]
        if c not in seq_acc_by_collision:
            seq_acc_by_collision[c] = {"correct": 0, "total": 0}
        seq_acc_by_collision[c]["total"] += 1
        if decoded == nxt:
            seq_acc_by_collision[c]["correct"] += 1

    print(f"  {'collisions':>10s}  {'facts':>8s}  {'sequences':>10s}")
    print(f"  {'─' * 10}  {'─' * 8}  {'─' * 10}")
    all_c = sorted(set(list(fact_acc_by_collision.keys()) + list(seq_acc_by_collision.keys())))
    for c in all_c:
        f = fact_acc_by_collision.get(c)
        s = seq_acc_by_collision.get(c)
        fp = f"{100*f['correct']/f['total']:>7.1f}%" if f else "   N/A   "
        sp = f"{100*s['correct']/s['total']:>7.1f}%" if s else "   N/A   "
        print(f"  {c:10d}  {fp:>8s}  {sp:>10s}")

    print("\nDone.")


if __name__ == "__main__":
    run()
