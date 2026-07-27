"""Tests for VSA relational reasoning prototype."""

from src.vsa.primitives import VSA
from src.vsa.relational import RelationalEncoder, RelationalMemory


def test_encode_decode_round_trip():
    vsa = VSA(dim=1000)
    encoder = RelationalEncoder(vsa)

    vec = encoder.encode_triple("cat", "chases", "mouse")
    s, r, o = encoder.decode_triple(vec)

    assert s == "cat", f"Expected cat, got {s}"
    assert r == "chases", f"Expected chases, got {r}"
    assert o == "mouse", f"Expected mouse, got {o}"


def test_partial_query():
    vsa = VSA(dim=1000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=1000)

    memory.store_triple("cat", "chases", "mouse")
    memory.store_triple("dog", "chases", "cat")
    memory.store_triple("mouse", "fears", "cat")

    s, r, o = memory.complete({"relation": "chases", "object": "mouse"})
    assert s == "cat", f"Expected cat, got {s}"

    s, r, o = memory.complete({"subject": "dog", "relation": "chases"})
    assert o == "cat", f"Expected cat, got {o}"

    s, r, o = memory.complete({"subject": "mouse", "object": "cat"})
    assert r == "fears", f"Expected fears, got {r}"


def test_multiple_triples_no_collision():
    vsa = VSA(dim=1000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=1000)

    triples = [
        ("cat", "chases", "mouse"),
        ("dog", "chases", "cat"),
        ("mouse", "fears", "cat"),
        ("bird", "chases", "worm"),
        ("fish", "swims", "water"),
    ]
    for s, r, o in triples:
        memory.store_triple(s, r, o)

    for s, r, o in triples:
        sr, rr, or_ = memory.complete({"subject": s, "relation": r})
        assert or_ == o, f"({s}, {r}, ?) expected {o}, got {or_}"

        sr, rr, or_ = memory.complete({"subject": s, "object": o})
        assert rr == r, f"({s}, ?, {o}) expected {r}, got {rr}"

        sr, rr, or_ = memory.complete({"relation": r, "object": o})
        assert sr == s, f"(?, {r}, {o}) expected {s}, got {sr}"


def test_analogy():
    vsa = VSA(dim=1000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=1000)

    triples = [
        ("cat", "chases", "mouse"),
        ("dog", "chases", "cat"),
        ("mouse", "fears", "cat"),
        ("bird", "fears", "cat"),
    ]
    for s, r, o in triples:
        memory.store_triple(s, r, o)

    # Analogy: (? chases mouse) should be cat (stored directly)
    s, r, o = memory.complete({"relation": "chases", "object": "mouse"})
    assert s == "cat"

    # Analogy: (dog ? cat) should be chases
    s, r, o = memory.complete({"subject": "dog", "object": "cat"})
    assert r == "chases"


def test_identity_two_triples():
    """Two triples with shared entity — verify no cross-talk."""
    vsa = VSA(dim=1000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=1000)

    memory.store_triple("cat", "chases", "mouse")
    memory.store_triple("cat", "fears", "dog")

    s, r, o = memory.complete({"subject": "cat", "relation": "chases"})
    assert o == "mouse", f"Expected mouse, got {o}"

    s, r, o = memory.complete({"subject": "cat", "relation": "fears"})
    assert o == "dog", f"Expected dog, got {o}"
