"""Test how Hopfield handles ambiguous continuations in 2-slot pairs,
and whether wider context context breaks chaining loops.

Three experiments:
  PART 1: Synthetic frequency-controlled collision test
  PART 2: Text collision + accuracy analysis at 1-token vs 2-token contexts
  PART 3: Chaining with N-token context (N=1,2,3)
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import torch

from src.vsa.primitives import VSA
from src.vsa.hopfield import HopfieldNet

# ── synthetic test ───────────────────────────────────────────────────

def synthetic_test(dim: int = 1000, modern_beta: float = 50.0):
    vsa = VSA(dim=dim)
    ir = vsa.make_vector()
    or_ = vsa.make_vector()

    scenarios = [
        ("A", [("T100", 100), ("T101", 50), ("T102", 1)]),
        ("B", [("T103", 90), ("T104", 10)]),
        ("C", [("T105", 70), ("T106", 30)]),
        ("D", [("T107", 50), ("T108", 50)]),
        ("E", [("T109", 40), ("T110", 30), ("T111", 20), ("T112", 10)]),
        ("F", [("T113", 20), ("T114", 20), ("T115", 20), ("T116", 20), ("T117", 20)]),
    ]

    left_tokens = [s[0] for s in scenarios]
    right_tokens = set()
    for _, rights in scenarios:
        for rt, _ in rights:
            right_tokens.add(rt)
    token_vecs = {t: vsa.make_vector() for t in left_tokens + list(right_tokens)}

    def encode_pair(left: str, right: str) -> torch.Tensor:
        return vsa.bundle([vsa.bind(ir, token_vecs[left]), vsa.bind(or_, token_vecs[right])])

    results = []
    for left, rights in scenarios:
        hop = HopfieldNet(dim, retrieval_mode="modern", modern_beta=modern_beta)
        total_stored = 0
        for right, count in rights:
            for _ in range(count):
                hop.store(encode_pair(left, right))
                total_stored += 1

        q = vsa.bind(ir, token_vecs[left])
        recalled = hop.recall(q)
        unbound = vsa.unbind(recalled, or_)
        all_names = list(token_vecs.keys())
        all_vecs = torch.stack([token_vecs[n] for n in all_names])
        decoded = all_names[vsa.similarity(unbound, all_vecs).argmax().item()]

        dominant = max(rights, key=lambda x: x[1])[0]
        dom_share = max(count for _, count in rights) / total_stored * 100
        correct = decoded == dominant
        results.append({
            "left": left, "n_rights": len(rights),
            "dominant_pct": round(dom_share, 1),
            "decoded": decoded, "correct": correct,
        })
        print(f"  {left:5s} | {len(rights):2d} rights | dominant={dom_share:5.1f}% | decoded={decoded:5s} | {'✓' if correct else '✗'}")
    return results


# ── text accuracy ────────────────────────────────────────────────────

def tokenize(text: str, chunk_width: int) -> list[str]:
    tokens = []
    for i in range(0, len(text), chunk_width):
        chunk = text[i:i + chunk_width].ljust(chunk_width, "_")
        tokens.append(chunk)
    return tokens


def build_patterns(tokens: list[str], context_width: int) -> list[tuple[list[str], str]]:
    """Build (context_tokens, next_token) patterns.

    context_width=1: (token_i → token_{i+1})
    context_width=2: (token_i, token_{i+1} → token_{i+2})
    """
    patterns = []
    for i in range(len(tokens) - context_width):
        ctx = [tokens[i + j] for j in range(context_width)]
        nxt = tokens[i + context_width]
        patterns.append((ctx, nxt))
    return patterns


def encode_ctx(ctx_tokens: list[str], roles: list[torch.Tensor], token_vecs: dict, vsa) -> torch.Tensor:
    """Encode context tokens as a bundled VSA vector, one role per position."""
    components = []
    for tok, role in zip(ctx_tokens, roles):
        components.append(vsa.bind(role, token_vecs[tok]))
    return vsa.bundle(components)


def text_accuracy(text: str, dim: int = 1000, chunk_width: int = 2, context_width: int = 1,
                  modern_beta: float = 50.0) -> dict:
    vsa = VSA(dim=dim)
    tokens = tokenize(text, chunk_width)
    patterns = build_patterns(tokens, context_width)

    # Role vectors: one per context position
    ctx_roles = [vsa.make_vector() for _ in range(context_width)]
    out_role = vsa.make_vector()

    all_tokens = sorted(set(tokens))
    token_vecs = {t: vsa.make_vector() for t in all_tokens}

    # Store patterns
    hop = HopfieldNet(dim, retrieval_mode="modern", modern_beta=modern_beta)
    for ctx, nxt in patterns:
        ctx_vec = encode_ctx(ctx, ctx_roles, token_vecs, vsa)
        pair_vec = vsa.bundle([ctx_vec, vsa.bind(out_role, token_vecs[nxt])])
        hop.store(pair_vec)

    # Query: for each pattern, check if recall produces the correct next token
    all_names = list(token_vecs.keys())
    all_vecs = torch.stack([token_vecs[n] for n in all_names])
    correct = 0
    for ctx, nxt in patterns:
        q_vec = encode_ctx(ctx, ctx_roles, token_vecs, vsa)
        recalled = hop.recall(q_vec)
        unbound = vsa.unbind(recalled, out_role)
        decoded = all_names[vsa.similarity(unbound, all_vecs).argmax().item()]
        if decoded == nxt:
            correct += 1

    acc = 100 * correct / len(patterns)
    return {"accuracy": round(acc, 1), "n_patterns": len(patterns), "unique_tokens": len(all_tokens)}


# ── chaining with N-token context ────────────────────────────────────

def chain(text: str, dim: int = 1000, chunk_width: int = 2, context_width: int = 1,
          max_steps: int = 30, modern_beta: float = 50.0) -> str:
    vsa = VSA(dim=dim)
    tokens = tokenize(text, chunk_width)
    patterns = build_patterns(tokens, context_width)

    ctx_roles = [vsa.make_vector() for _ in range(context_width)]
    out_role = vsa.make_vector()

    all_tokens = sorted(set(tokens))
    token_vecs = {t: vsa.make_vector() for t in all_tokens}
    all_names = list(token_vecs.keys())
    all_vecs = torch.stack([token_vecs[n] for n in all_names])

    hop = HopfieldNet(dim, retrieval_mode="modern", modern_beta=modern_beta)
    for ctx, nxt in patterns:
        ctx_vec = encode_ctx(ctx, ctx_roles, token_vecs, vsa)
        hop.store(vsa.bundle([ctx_vec, vsa.bind(out_role, token_vecs[nxt])]))

    # Seed: first context_width tokens from corpus
    current_ctx = list(tokens[:context_width])
    generated = list(current_ctx)

    for step in range(max_steps):
        q_vec = encode_ctx(current_ctx, ctx_roles, token_vecs, vsa)
        recalled = hop.recall(q_vec)
        unbound = vsa.unbind(recalled, out_role)
        decoded = all_names[vsa.similarity(unbound, all_vecs).argmax().item()]
        generated.append(decoded)
        # Slide the context window
        current_ctx = current_ctx[1:] + [decoded]

    raw = "".join(generated).replace("_", " ")
    return raw


# ── main ─────────────────────────────────────────────────────────────

CORPORA = {
    "short (50c)": "the cat sat on the mat and the dog ran in the park",
    "medium (186c)": (
        "the quick brown fox jumps over the lazy dog near the bank of the river "
        "while the small grey mouse runs through the green field to find some "
        "cheese and nuts for the long cold winter ahead"
    ),
    "longer (202c)": (
        "in the beginning there was the word and the word was with god and the "
        "word was god the same was in the beginning with god all things were made "
        "by him and without him was not any thing made that was made"
    ),
}


def run():
    # ── PART 1 ──
    print("PART 1: SYNTHETIC COLLISION TEST")
    print("=" * 50)
    synthetic_test()

    # ── PART 2 ──
    print("\n\nPART 2: ACCURACY vs CONTEXT WIDTH")
    print("=" * 50)
    header = f"{'corpus':20s} {'cw':3s} {'ctx':3s} {'accuracy':>8s} {'patterns':>9s} {'tokens':>7s}"
    print(header)
    print("-" * len(header))
    for name, text in CORPORA.items():
        for cw in [1, 2]:
            for ctx in [1, 2, 3]:
                r = text_accuracy(text, chunk_width=cw, context_width=ctx)
                print(f"{name:20s} {cw:3d} {ctx:3d} {r['accuracy']:>7.1f}% {r['n_patterns']:>9d} {r['unique_tokens']:>7d}")

    # ── PART 3 ──
    print("\n\nPART 3: CHAINING WITH N-TOKEN CONTEXT")
    print("=" * 50)
    for name, text in CORPORA.items():
        print(f"\n── {name} ──")
        for cw in [2]:
            for ctx in [1, 2, 3]:
                raw = chain(text, chunk_width=cw, context_width=ctx, max_steps=30)
                print(f"  cw={cw} ctx={ctx}: {raw}")


if __name__ == "__main__":
    run()
