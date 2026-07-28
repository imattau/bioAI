"""What RelationalMemory already replicates from ConsolidationMemory's
relational half, and what it genuinely doesn't yet.

ConsolidationMemory (src/text/consolidation.py) does two unrelated jobs:
concept-prototype formation for semantic retrieval scoring (untouched here,
has no equivalent in RelationalMemory, not a candidate for removal), and
relation extraction/tie-breaking/path-reasoning (relations, relation_claims,
relation_candidates, reason_path) — the part that overlaps with
RelationalMemory + resolve().

This file mirrors the exact scenarios from tests/test_text/test_consolidation.py
through RelationalMemory instead, to get an evidence-based answer to "what
can be removed" rather than a design-intuition one. Two capabilities turn
out to be genuinely missing — and unreliable enough (dependent on incidental
random-vector geometry rather than the actual facts stored) that a single
assertion flaked pass/fail on an unseeded draw during development, which is
itself part of the finding. Both are tested here as a multi-trial statistical
claim instead: "never reliably correct across many independent substrates,"
which is robust and won't itself flake. Until both are addressed,
ConsolidationMemory's relational half must stay.
"""

from src.vsa.primitives import VSA
from src.vsa.relational import RelationalEncoder, RelationalMemory


def _memory(dim: int = 2000) -> RelationalMemory:
    return RelationalMemory(RelationalEncoder(VSA(dim=dim)), dim=dim)


# ── Replicated: tie abstention ──────────────────────────────────────────
# Mirrors test_reasoning_abstains_on_tied_conflict. ConsolidationMemory's
# _supported_choice returns None on an exact count tie and reason_path
# abstains; RelationalMemory's ground-truth ambiguity check does the same
# job, and additionally names the tied candidates instead of a bare None.

def test_replicates_tie_abstention():
    memory = _memory()
    memory.store_triple("springfield", "in", "illinois")
    memory.store_triple("springfield", "in", "massachusetts")

    result = memory.complete_detailed(
        {"subject": "springfield", "relation": "in"}, top_k=3
    )
    assert result.ambiguous
    assert {name for name, _ in result.candidates} >= {"illinois", "massachusetts"}


# ── Replicated: confident single-fact retrieval ─────────────────────────
# Mirrors test_agent_uses_consolidated_relation_evidence's non-conflicting case.

def test_replicates_unambiguous_single_fact_retrieval():
    memory = _memory()
    memory.store_triple("france", "capital", "paris")

    result = memory.complete_detailed(
        {"subject": "france", "relation": "capital"}, top_k=3
    )
    assert result.best == "paris"
    assert not result.ambiguous


# ── NOT (yet) replicated: frequency-weighted evidence ───────────────────
# Mirrors test_relation_conflicts_are_counted_not_overwritten: Canberra
# asserted twice, Sydney once, should rank Canberra first. ConsolidationMemory
# tracks this with an explicit integer count per claim. RelationalMemory has
# no equivalent — storing the same triple twice just appends two identical
# vectors to the same Hopfield net, and at the sharp beta used for confident
# single-fact recall, whichever single pattern has a fractionally higher raw
# similarity score wins the softmax outright; duplicate entries don't reliably
# accumulate weight the way an explicit count does. A single draw of this
# assertion flaked between pass and fail depending on encoding noise, which
# is itself the finding: a real evidence-count mechanism gets the majority
# claim right on every substrate, not depending on incidental vector
# geometry. So this runs the scenario across many independent random VSA
# instances and asserts the majority claim does NOT reliably win.
def test_does_not_yet_replicate_evidence_weighted_conflicts():
    successes = 0
    trials = 30
    for _ in range(trials):
        memory = _memory(dim=500)
        memory.store_triple("australia", "capital", "canberra")
        memory.store_triple("australia", "capital", "canberra")
        memory.store_triple("australia", "capital", "sydney")

        result = memory.complete_detailed(
            {"subject": "australia", "relation": "capital"}, top_k=3
        )
        successes += result.best == "canberra"

    # If this is ever unanimous, RelationalMemory has gained reliable
    # frequency weighting and this gap should be reassessed, not silently
    # left stale.
    assert successes < trials, (
        f"{successes}/{trials} — if this is ever unanimous, evidence-count "
        f"weighting may have been added and this test should be reassessed"
    )


# ── NOT (yet) replicated: transitive multi-hop chaining ─────────────────
# Mirrors test_agent_composes_two_hop_answer_with_evidence: reason_path
# derives "canberra is in oceania" by chaining capital_of -> in edges through
# an intermediate unknown (australia). This is a structurally different
# operation from resolve()'s intersection of independent evidence about one
# unknown — there is no unknown-chaining in RelationalMemory at all. A naive
# query for (canberra, in, ?) can *look* like it works when the "in" net
# happens to hold only one pattern (Hopfield degenerates to returning that
# pattern regardless of query content, a sparsity artifact) or even with a
# decoy present, purely by chance correlation between canberra's vector and
# oceania's — a single run of this assertion flaked from fail to pass
# (XPASS) on an unseeded random draw, which is itself the proof: a real
# chaining mechanism would get this right on every substrate, not depending
# on incidental vector geometry. So instead of one draw, this runs the same
# scenario across many independent random VSA instances and asserts the
# answer is NOT reliably correct.
def test_does_not_yet_replicate_transitive_chaining():
    successes = 0
    trials = 30
    for _ in range(trials):
        memory = _memory(dim=500)  # smaller dim: more incidental correlation, not less
        memory.store_triple("canberra", "capital_of", "australia")
        memory.store_triple("australia", "in", "oceania")
        memory.store_triple("france", "in", "europe")  # decoy: forces a real choice

        result = memory.complete_detailed(
            {"subject": "canberra", "relation": "in"}, top_k=3
        )
        successes += result.best == "oceania"

    # A real chaining mechanism gets this right every time, regardless of the
    # random vector substrate. If RelationalMemory ever reaches ~30/30 here,
    # that's a sign real chaining exists and this test should be revisited —
    # until then, anything short of unanimous success across independent
    # substrates confirms it's coincidence, not reasoning.
    assert successes < trials, (
        f"{successes}/{trials} — if this is ever unanimous, transitive "
        f"chaining may have been added and this test should be reassessed"
    )


# ── Now available: persistence ────────────────────────────────────────
# Mirrors test_relations_survive_agent_save_load. This gap is closed —
# RelationalMemory/RelationalEncoder now have get_state/from_state (see
# tests/test_vsa/test_relational.py::test_persistence_round_trip for the
# full round-trip check), and BioAIDialogueAgent.save/load persists
# self.relational. Kept here (rather than deleted) as the corresponding
# "replicated" entry to the original gap this file documented.

def test_persistence_now_available():
    memory = _memory()
    memory.store_triple("france", "capital", "paris")
    restored = RelationalMemory.from_state(memory.get_state())
    assert restored.complete_detailed(
        {"subject": "france", "relation": "capital"}
    ).best == "paris"
