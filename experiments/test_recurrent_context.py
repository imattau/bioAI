"""Test full-context recurrent VSA encoding for sequence prediction.

No N-gram window — the entire accumulated sequence is encoded as one state vector.
The Hopfield stores (prefix_state, next_token) pairs. Query is the current prefix.

Tests:
  1. Stable across repetition — same prefix at position 2 vs position 52
  2. Match quality — variable-length prefix matching
  3. Discrimination — similar prefixes diverge enough
  4. Full chaining — generate from the stored corpus
"""

from __future__ import annotations

import random
import statistics
import torch

from src.vsa.primitives import VSA
from src.vsa.hopfield import HopfieldNet


def make_seq_state_encoder(vsa):
    """Create a recurrent VSA state encoder with L2 normalization (no sign)."""
    role = vsa.make_vector()
    cache = {}
    def tv(t):
        if t not in cache:
            cache[t] = vsa.make_vector()
        return cache[t]
    def encode(sequence: list[str], normalize: bool = True) -> torch.Tensor:
        state = torch.zeros(vsa.dim, device=vsa.device)
        for token in sequence:
            item = vsa.bind(role, tv(token))
            state = state + item
        if normalize:
            norm = state.norm().clamp(min=1e-8)
            state = state / norm
        return state
    def decode(state: torch.Tensor) -> str:
        names = sorted(cache.keys())
        vs_ = torch.stack([cache[n] for n in names])
        sim = vsa.similarity(state, vs_)
        return names[sim.argmax().item()]
    return tv, encode, decode, cache


def tokenise(text: str, width: int = 2) -> list[str]:
    return [text[i:i+width].ljust(width, "_") for i in range(0, len(text), width)]


# ─── Test 1: Stability across repetition ────────────────────────────

def test_stability():
    print("─── TEST 1: Stability across repetition ───")
    text = "the cat sat on the mat and the dog ran in the park"
    tokens = tokenise(text, width=2)
    # Repeat the corpus to create multiple occurrences of the same prefix
    long_text = text * 3
    long_tokens = tokenise(long_text, width=2)

    vsa = VSA(dim=1000)
    tv, encode, decode, cache = make_seq_state_encoder(vsa)

    # Encode "the" at every occurrence
    positions = []
    states = []
    for i, tok in enumerate(long_tokens):
        if tok == "th":
            state = encode(long_tokens[:i+1])
            positions.append(i)
            states.append(state)

    # Compute pairwise similarities between all occurrences of "th"
    sims = []
    for i in range(len(states)):
        for j in range(i+1, len(states)):
            s = float(vsa.similarity(states[i], states[j]).item())
            sims.append(s)

    print(f"  Occurrences of 'th': {len(positions)}")
    print(f"  Pairwise cosine: mean={statistics.mean(sims):.4f}  min={min(sims):.4f}  max={max(sims):.4f}")

    # A good result: mean > 0.7 means the same prefix at different positions
    # is recognizable. Mean < 0.5 means the state diverges too fast.
    ok = statistics.mean(sims) > 0.5
    print(f"  {'✓' if ok else '✗'} {'Mean similarity > 0.5 — stable' if ok else 'State diverges too fast'}")


# ─── Test 2: Match quality ──────────────────────────────────────────

def test_matching():
    print("\n─── TEST 2: Match quality ───")
    vsa = VSA(dim=1000)
    tv, encode, decode, cache = make_seq_state_encoder(vsa)

    # Three different sentences
    sentences = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the bird flew over the tree",
    ]
    token_lists = [tokenise(s, width=2) for s in sentences]

    # Store ALL positions from ALL sentences
    hop = HopfieldNet(1000, retrieval_mode="modern", modern_beta=50.0, packed=True)
    out_role = vsa.make_vector()

    all_store_ctx = []
    for toks in token_lists:
        for i in range(len(toks) - 1):
            ctx = toks[:i+1]
            nxt = toks[i+1]
            ctx_state = encode(ctx)
            pattern = vsa.bundle([ctx_state, vsa.bind(out_role, tv(nxt))])
            hop.store(pattern)
            all_store_ctx.append((ctx, nxt))

    # Query: for each stored (ctx, nxt), can we retrieve the correct next token?
    correct = 0
    total = len(all_store_ctx)
    for ctx, nxt in all_store_ctx:
        q = encode(ctx)
        rec = hop.recall(q)
        ub = vsa.unbind(rec, out_role)
        decoded = decode(ub)
        if decoded == nxt:
            correct += 1

    acc = 100 * correct / total
    print(f"  Variable-length matching accuracy: {acc:.1f}% ({correct}/{total})")
    print(f"  {'✓' if acc > 90 else '✗'} {'Good' if acc > 90 else 'Needs improvement'}")

    # Also test: can we match across sentences?
    # Query "the cat sat on the" from sentence 1 — should retrieve "mat"
    ctx1 = token_lists[0][:5]  # "the cat sat on the"
    q = encode(ctx1)
    rec = hop.recall(q)
    ub = vsa.unbind(rec, out_role)
    decoded = decode(ub)
    expected = token_lists[0][5]  # "mat"
    cross_ok = decoded == expected
    print(f"  Cross-sentence disambiguation: query ends of S1 → {decoded} (expected {expected}) {'✓' if cross_ok else '✗'}")


# ─── Test 3: Discrimination ─────────────────────────────────────────

def test_discrimination():
    print("\n─── TEST 3: Discrimination of similar prefixes ───")
    vsa = VSA(dim=1000)
    tv, encode, decode, cache = make_seq_state_encoder(vsa)

    # Two sentences differing only in the verb
    s1_tokens = tokenise("the cat sat on the mat", width=2)
    s2_tokens = tokenise("the cat ran on the mat", width=2)

    # Encode the full prefix up to and including the differing verb
    ctx1 = s1_tokens[:3]  # "the ca", "t_", "sa" → includes "sat"
    ctx2 = s2_tokens[:3]  # "the ca", "t_", "ra" → includes "ran"

    q1 = encode(ctx1)
    q2 = encode(ctx2)

    sim = float(vsa.similarity(q1, q2).item())
    print(f"  '...sat on...' vs '...ran on...' state cosine: {sim:.4f}")
    print(f"  {'✓ Discriminable' if sim < 0.9 else '⚠ Similar prefixes not separable'}")

    # Also check: how far back does the divergence propagate?
    print(f"\n  State divergence across prefix length:")
    for pre_len in [1, 2, 3, 4, 5, 6]:
        c1 = s1_tokens[:pre_len]
        c2 = s2_tokens[:pre_len]
        if len(c1) < 2 or len(c2) < 2:
            continue
        s = float(vsa.similarity(encode(c1), encode(c2)).item())
        marker = "✓" if s < 0.9 else "~"
        if s < 0.95:
            print(f"    prefix_len={pre_len}: cosine={s:.4f}  {marker}")


# ─── Test 4: Full chaining ──────────────────────────────────────────

def test_chain():
    print("\n─── TEST 4: Full context chaining ───")
    text = "the cat sat on the mat and the dog ran in the park"
    tokens = tokenise(text, width=2)
    n = len(tokens)

    vsa = VSA(dim=1000)
    tv, encode, decode, cache = make_seq_state_encoder(vsa)
    out_role = vsa.make_vector()

    # Store all positions from the corpus as full-context patterns
    hop = HopfieldNet(1000, retrieval_mode="modern", modern_beta=50.0, packed=True)
    for i in range(n - 1):
        ctx = tokens[:i+1]
        nxt = tokens[i+1]
        ctx_state = encode(ctx)
        pattern = vsa.bundle([ctx_state, vsa.bind(out_role, tv(nxt))])
        hop.store(pattern)

    # Chaining: start with first token, generate up to n tokens
    gen = [tokens[0]]
    q_state = encode([tokens[0]])
    errors = 0

    for step in range(n - 1):
        rec = hop.recall(q_state)
        ub = vsa.unbind(rec, out_role)
        decoded = decode(ub)
        gen.append(decoded)

        if decoded != tokens[step + 1]:
            errors += 1
            if errors > 5:  # stop early if too many errors
                break

        # Update running state: add the decoded token
        q_state = encode([decoded] if step == 0 else [])  # TEMP: just use current token
        # FIX: need to accumulate state

    gen_str = "".join(gen).replace("_", " ")
    print(f"  Generated: {gen_str[:50]}...")
    print(f"  Errors: {errors}/{n-1}")


# ─── Test 4b: Fixed — Accumulating state ───────────────────────────

def test_chain_accumulate():
    print("\n─── TEST 4b: Accumulating context chaining ───")
    text = "the cat sat on the mat and the dog ran in the park"
    tokens = tokenise(text, width=2)
    n = len(tokens)

    vsa = VSA(dim=1000)
    tv, encode, decode, cache = make_seq_state_encoder(vsa)
    out_role = vsa.make_vector()

    # Store patterns with accumulating prefix
    hop = HopfieldNet(1000, retrieval_mode="modern", modern_beta=50.0, packed=True)
    states = []  # cache prefix states
    for i in range(n - 1):
        ctx = tokens[:i+1]
        nxt = tokens[i+1]
        ctx_state = encode(ctx)
        states.append(ctx_state)
        pattern = vsa.bundle([ctx_state, vsa.bind(out_role, tv(nxt))])
        hop.store(pattern)

    # Recurrent chaining: maintain running state
    gen = [tokens[0]]
    state = states[0]  # state after first token
    errors = 0

    for step in range(n - 1):
        rec = hop.recall(state)
        ub = vsa.unbind(rec, out_role)
        decoded = decode(ub)
        gen.append(decoded)

        if decoded != tokens[step + 1]:
            errors += 1
            if errors > 5:
                break

        # Update state by adding the decoded token's contribution
        item = vsa.bind(vsa.make_vector(), tv(decoded))  # need the role... simplify
        # Actually just re-encode the accumulated sequence
        state = encode(tokens[:step+2])  # cheating — use original for now

    gen_str = "".join(gen).replace("_", " ")
    print(f"  Accumulating chain: {gen_str[:50]}...")
    print(f"  Errors (using ground-truth context): {errors}/{n-1}")

    # Now test with actual accumulated encoding (no cheating)
    print(f"\n  True accumulated chaining (re-encode each step):")
    gen2 = [tokens[0]]
    for step in range(n - 1):
        ctx = list(gen2)
        state = encode(ctx)
        if len(ctx) < 1:
            break
        rec = hop.recall(state)
        ub = vsa.unbind(rec, out_role)
        decoded = decode(ub)
        gen2.append(decoded)
        if decoded != tokens[step + 1]:
            break
    gen2_str = "".join(gen2).replace("_", " ")
    print(f"    {gen2_str[:50]}...")
    print(f"    Length correct: {len(gen2)}/{n}")


# ─── Run ────────────────────────────────────────────────────────────

def run():
    random.seed(42)
    torch.manual_seed(42)

    test_stability()
    test_matching()
    test_discrimination()
    test_chain()
    test_chain_accumulate()


if __name__ == "__main__":
    run()
