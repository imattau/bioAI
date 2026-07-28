"""Tests for VSA relational reasoning prototype."""

import pytest

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


def test_genuine_collision_is_flagged_ambiguous():
    """When a partial cue truly matches >1 stored triple, no decode can be
    'correct' — the memory should surface this instead of silently guessing.
    """
    vsa = VSA(dim=1000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=1000)

    memory.store_triple("cat", "chases", "mouse")
    memory.store_triple("dog", "chases", "mouse")  # same (relation, object), different subject

    matches = memory.ground_truth_ambiguity({"relation": "chases", "object": "mouse"})
    assert {s for s, _, _ in matches} == {"cat", "dog"}

    result = memory.complete_detailed({"relation": "chases", "object": "mouse"}, top_k=2)
    assert result.ambiguous
    assert {name for name, _ in result.candidates} == {"cat", "dog"}


def test_unique_query_not_flagged_ambiguous():
    vsa = VSA(dim=1000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=1000)

    memory.store_triple("cat", "chases", "mouse")
    memory.store_triple("dog", "chases", "cat")

    assert len(memory.ground_truth_ambiguity({"relation": "chases", "object": "mouse"})) == 1
    result = memory.complete_detailed({"relation": "chases", "object": "mouse"})
    assert result.best == "cat"
    assert not result.ambiguous


def test_context_disambiguates_collision():
    """A (relation, object) collision that's ambiguous on its own becomes
    resolvable once a discriminating context role is bound into both
    storage and query — analogous to more context narrowing an LLM's
    next-token distribution.
    """
    vsa = VSA(dim=2000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=2000)

    memory.store_triple("cat", "chases", "mouse", context={"scene": "kitchen"})
    memory.store_triple("dog", "chases", "mouse", context={"scene": "garden"})

    # Without context: genuinely ambiguous, two stored matches.
    known = {"relation": "chases", "object": "mouse"}
    assert len(memory.ground_truth_ambiguity(known)) == 2
    assert memory.complete_detailed(known).ambiguous

    # With the right context: uniquely determined.
    assert memory.ground_truth_ambiguity(known, context={"scene": "kitchen"}) == [
        ("cat", "chases", "mouse")
    ]
    result = memory.complete_detailed(known, context={"scene": "kitchen"})
    assert result.best == "cat"
    assert not result.ambiguous

    result = memory.complete_detailed(known, context={"scene": "garden"})
    assert result.best == "dog"
    assert not result.ambiguous


def test_context_does_not_help_when_shared():
    """Context only disambiguates when it actually differs between the
    colliding triples — no amount of conditioning manufactures information
    that was never captured.
    """
    vsa = VSA(dim=2000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=2000)

    memory.store_triple("cat", "chases", "mouse", context={"scene": "kitchen"})
    memory.store_triple("dog", "chases", "mouse", context={"scene": "kitchen"})

    known = {"relation": "chases", "object": "mouse"}
    matches = memory.ground_truth_ambiguity(known, context={"scene": "kitchen"})
    assert {s for s, _, _ in matches} == {"cat", "dog"}
    assert memory.complete_detailed(known, context={"scene": "kitchen"}).ambiguous


def test_store_and_complete_without_context_unaffected():
    """Context is opt-in — omitting it entirely must behave exactly as before."""
    vsa = VSA(dim=1000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=1000)

    memory.store_triple("cat", "chases", "mouse")
    s, r, o = memory.complete({"relation": "chases", "object": "mouse"})
    assert s == "cat"


def test_resolve_intersects_candidates_across_steps():
    """Two independent clues about the same unknown subject narrow the
    candidate pool via intersection, resolving a collision neither clue
    resolves alone.
    """
    vsa = VSA(dim=2000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=2000)

    memory.store_triple("cat", "chases", "mouse")
    memory.store_triple("dog", "chases", "mouse")   # collision: {cat, dog}
    memory.store_triple("dog", "fears", "water")
    memory.store_triple("fox", "fears", "water")    # collision: {dog, fox}

    trace = memory.resolve([
        ({"relation": "chases", "object": "mouse"}, None),
        ({"relation": "fears", "object": "water"}, None),
    ], top_k=2)

    assert trace.resolved
    assert trace.final_candidates == ["dog"]
    assert not trace.contradictory
    assert trace.steps[0].survivors == ["cat", "dog"]
    assert trace.steps[0].informative
    assert trace.steps[1].survivors == ["dog"]
    assert trace.steps[1].informative


def test_resolve_marks_uninformative_step():
    """A step whose candidates don't narrow the pool must not be reported
    as having contributed to the resolution.
    """
    vsa = VSA(dim=2000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=2000)

    memory.store_triple("cat", "chases", "mouse")
    memory.store_triple("dog", "chases", "mouse")
    memory.store_triple("cat", "hides-from", "owl")
    memory.store_triple("dog", "hides-from", "owl")  # same {cat, dog} — no new info
    memory.store_triple("dog", "fears", "water")
    memory.store_triple("fox", "fears", "water")

    trace = memory.resolve([
        ({"relation": "chases", "object": "mouse"}, None),
        ({"relation": "hides-from", "object": "owl"}, None),
        ({"relation": "fears", "object": "water"}, None),
    ], top_k=2)

    assert len(trace.steps) == 3
    assert trace.steps[0].informative
    assert not trace.steps[1].informative
    assert trace.steps[2].informative
    assert trace.resolved
    assert trace.final_candidates == ["dog"]


def test_resolve_flags_contradiction_without_discarding_progress():
    """Evidence that conflicts with an already-narrowed pool is flagged,
    not silently accepted or allowed to erase prior progress — including
    when it arrives after the pool has already resolved to one candidate.
    """
    vsa = VSA(dim=2000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=2000)

    memory.store_triple("cat", "chases", "mouse")
    memory.store_triple("dog", "chases", "mouse")
    memory.store_triple("dog", "fears", "water")
    memory.store_triple("fox", "fears", "water")
    memory.store_triple("cat", "likes", "fish")     # conflicts with "dog"
    memory.store_triple("mouse", "likes", "fish")

    trace = memory.resolve([
        ({"relation": "chases", "object": "mouse"}, None),
        ({"relation": "fears", "object": "water"}, None),   # narrows to {dog}
        ({"relation": "likes", "object": "fish"}, None),    # conflicts
    ], top_k=2)

    assert trace.resolved
    assert trace.final_candidates == ["dog"]
    assert trace.contradictory
    assert trace.steps[-1].contradictory
    assert trace.steps[-1].survivors == ["dog"]  # prior pool kept, not wiped


def test_resolve_requires_same_target_role():
    vsa = VSA(dim=1000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=1000)
    memory.store_triple("cat", "chases", "mouse")

    with pytest.raises(ValueError):
        memory.resolve([
            ({"relation": "chases", "object": "mouse"}, None),  # targets "subject"
            ({"subject": "cat", "object": "mouse"}, None),       # targets "relation"
        ])


def test_resolve_requires_at_least_one_query():
    vsa = VSA(dim=1000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=1000)

    with pytest.raises(ValueError):
        memory.resolve([])


def test_persistence_round_trip():
    vsa = VSA(dim=1000)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=1000)

    memory.store_triple("cat", "chases", "mouse")
    memory.store_triple("dog", "chases", "mouse", context={"scene": "garden"})
    memory.store_triple("dog", "fears", "water")

    restored = RelationalMemory.from_state(memory.get_state())

    # Same completions, including ambiguity, after restore.
    before = memory.complete_detailed({"relation": "chases", "object": "mouse"})
    after = restored.complete_detailed({"relation": "chases", "object": "mouse"})
    assert after.ambiguous == before.ambiguous
    assert set(after.candidates) == set(before.candidates)

    # ground_truth_ambiguity / context still work post-restore.
    assert restored.ground_truth_ambiguity(
        {"relation": "chases", "object": "mouse"}, context={"scene": "garden"}
    ) == [("dog", "chases", "mouse")]

    # A fact stored before restore is still confidently retrievable.
    result = restored.complete_detailed({"subject": "dog", "relation": "fears"})
    assert result.best == "water"
    assert not result.ambiguous

    # New facts can still be stored and queried after restore (nets/store
    # aren't frozen read-only copies).
    restored.store_triple("fox", "fears", "water")
    result = restored.complete_detailed({"subject": "fox", "relation": "fears"})
    assert result.best == "water"
