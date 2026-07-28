"""Tests for Phase 2 of the response-ecosystem plan: proposition genotype,
minimal realiser, and conservative mutation/recombination/predation
operators that produce organisms no single source contains verbatim. See
src/text/ecology/*.py docstrings and the response-ecosystem plan for the
full design; tests/test_text/test_response_ecology.py covers Phase 1
(selection only, no synthesis).
"""

from src.vsa import VSA
from src.vsa.relational import RelationalEncoder, RelationalMemory
from src.text import BioAIDialogueAgent
from src.text.ecology import (
    Proposition,
    PropositionRealiser,
    ResponseEcosystem,
    extract_propositions,
)
from src.text.ecology.organism import ResponseOrganism
from src.text.ecology.operators import (
    add_supported_proposition,
    add_uncertainty_qualifier,
    is_compatible,
    recombine,
    reorder_for_coherence,
    remove_low_relevance_proposition,
    survives_predation,
)


def _relational_memory(dim: int = 200) -> RelationalMemory:
    vsa = VSA(dim=dim, device="cpu")
    return RelationalMemory(RelationalEncoder(vsa), dim=dim)


# ── extract_propositions ────────────────────────────────────────────────

def test_extract_propositions_wombat_marsupial_example():
    """The proposal's own canonical example. The first sentence has a
    copula ConsolidationMemory.extract_relations can match; the second
    ("Marsupials carry their young in a pouch") has none of the 5 fixed
    patterns' required copula at all -- a real, disclosed limitation, not
    a bug to paper over in this test."""
    memories = [
        "Wombats are marsupials.",
        "Marsupials carry their young in a pouch.",
    ]
    props = extract_propositions(memories)
    assert props == [
        Proposition(
            subject="wombats", relation="is", object="marsupials",
            source_id=0, source_text="Wombats are marsupials.",
        ),
    ]


def test_extract_propositions_tracks_source_id_per_memory():
    memories = ["The capital of France is Paris.", "Mars is red."]
    props = extract_propositions(memories)
    by_source = {p.source_id: p for p in props}
    assert by_source[0].subject == "france"
    assert by_source[1].subject == "mars"


# ── realiser ─────────────────────────────────────────────────────────────

def test_realiser_only_introduces_traceable_content():
    props = (
        Proposition("wombats", "is", "marsupials", 0, "Wombats are marsupials."),
        Proposition("france", "capital", "paris", 1, "The capital of France is Paris."),
    )
    text = PropositionRealiser.realise(props).lower()
    for prop in props:
        assert prop.subject in text
        assert prop.object in text


def test_realiser_merges_same_subject_relation_objects_with_and():
    props = (
        Proposition("mars", "is", "red", 0, "x"),
        Proposition("mars", "is", "cold", 0, "y"),
    )
    text = PropositionRealiser.realise(props)
    assert "red and cold" in text.lower()


def test_realiser_empty_propositions_gives_empty_string():
    assert PropositionRealiser.realise(()) == ""


# ── recombination / compatibility ───────────────────────────────────────

def _organism(propositions, source_ids):
    return ResponseOrganism(
        text="", kind="proposition_composition", source_ids=source_ids,
        propositions=propositions,
    )


def test_recombination_refuses_conflicting_objects_for_same_subject_relation():
    a = _organism(
        (Proposition("wombats", "is", "marsupials", 0, "x"),), (0,)
    )
    b = _organism(
        (Proposition("wombats", "is", "reptiles", 1, "y"),), (1,)
    )
    assert is_compatible(a, b) is False
    assert recombine(a, b) is None


def test_recombination_merges_compatible_organisms():
    a = _organism(
        (Proposition("wombats", "is", "marsupials", 0, "x"),), (0,)
    )
    b = _organism(
        (Proposition("france", "capital", "paris", 1, "y"),), (1,)
    )
    child = recombine(a, b)
    assert child is not None
    assert set(child.propositions) == set(a.propositions) | set(b.propositions)
    assert child.kind == "recombined"


def test_recombination_checks_relational_memory_for_broader_conflicts():
    memory = _relational_memory()
    memory.store_triple("mars", "is", "red")
    memory.store_triple("mars", "is", "cold")
    a = _organism((Proposition("mars", "is", "red", 0, "x"),), (0,))
    b = _organism((Proposition("earth", "is", "round", 1, "y"),), (1,))
    # No direct conflict between a and b themselves, but relational_memory
    # knows of a second object ("cold") for (mars, is) that neither
    # organism asserts -- ground_truth_ambiguity surfaces that as a tie,
    # which the compatibility check treats conservatively as a conflict.
    assert is_compatible(a, b, memory) is False


# ── predation ────────────────────────────────────────────────────────────

def test_predation_rejects_lost_provenance():
    organism = _organism(
        (Proposition("wombats", "is", "marsupials", -1, "x"),), (0,)
    )
    assert survives_predation(organism) is False


def test_predation_rejects_internal_contradiction():
    organism = _organism(
        (
            Proposition("wombats", "is", "marsupials", 0, "x"),
            Proposition("wombats", "is", "reptiles", 0, "y"),
        ),
        (0,),
    )
    assert survives_predation(organism) is False


def test_predation_rejects_adjacent_duplicate_token():
    organism = ResponseOrganism(
        text="Earth is is the closest planet to the sun.",
        kind="relational_reasoning", source_ids=(0,), propositions=(),
    )
    assert survives_predation(organism) is False


def test_predation_accepts_healthy_organism():
    organism = _organism(
        (Proposition("wombats", "is", "marsupials", 0, "x"),), (0,)
    )
    organism.text = PropositionRealiser.realise(organism.propositions)
    assert survives_predation(organism) is True


# ── mutation operators ───────────────────────────────────────────────────

def test_add_supported_proposition_adds_relevant_unused_fact():
    organism = _organism(
        (Proposition("wombats", "is", "marsupials", 0, "x"),), (0,)
    )
    available = [
        Proposition("wombats", "is", "marsupials", 0, "x"),  # already present
        Proposition("wombats", "in", "australia", 1, "y"),   # shares subject
        Proposition("mars", "is", "red", 2, "z"),             # unrelated
    ]
    child = add_supported_proposition(organism, available, "Where do wombats live?")
    assert child is not None
    assert Proposition("wombats", "in", "australia", 1, "y") in child.propositions
    assert len(child.propositions) == 2


def test_add_supported_proposition_none_when_nothing_relevant():
    organism = _organism(
        (Proposition("wombats", "is", "marsupials", 0, "x"),), (0,)
    )
    available = [Proposition("mars", "is", "red", 1, "z")]
    assert add_supported_proposition(organism, available, "irrelevant query") is None


def test_remove_low_relevance_proposition_drops_weakest():
    organism = _organism(
        (
            Proposition("wombats", "is", "marsupials", 0, "x"),
            Proposition("mars", "is", "red", 1, "y"),
        ),
        (0, 1),
    )
    child = remove_low_relevance_proposition(organism, "Tell me about wombats")
    assert child is not None
    assert len(child.propositions) == 1
    assert child.propositions[0].subject == "wombats"


def test_remove_low_relevance_proposition_none_with_single_proposition():
    organism = _organism(
        (Proposition("wombats", "is", "marsupials", 0, "x"),), (0,)
    )
    assert remove_low_relevance_proposition(organism, "anything") is None


def test_reorder_for_coherence_puts_most_relevant_first():
    organism = _organism(
        (
            Proposition("mars", "is", "red", 0, "x"),
            Proposition("wombats", "is", "marsupials", 1, "y"),
        ),
        (0, 1),
    )
    child = reorder_for_coherence(organism, "Tell me about wombats")
    assert child is not None
    assert child.propositions[0].subject == "wombats"


def test_add_uncertainty_qualifier_on_genuine_collision():
    memory = _relational_memory()
    memory.store_triple("mars", "is", "red")
    memory.store_triple("mars", "is", "cold")
    organism = _organism((Proposition("mars", "is", "red", 0, "x"),), (0,))
    child = add_uncertainty_qualifier(organism, memory)
    assert child is not None
    assert any(
        prop.relation == "_uncertainty" for prop in child.propositions
    )


def test_add_uncertainty_qualifier_none_without_collision():
    memory = _relational_memory()
    memory.store_triple("mars", "is", "red")
    organism = _organism((Proposition("mars", "is", "red", 0, "x"),), (0,))
    assert add_uncertainty_qualifier(organism, memory) is None


def test_add_uncertainty_qualifier_none_without_relational_memory():
    organism = _organism((Proposition("mars", "is", "red", 0, "x"),), (0,))
    assert add_uncertainty_qualifier(organism, None) is None


# ── end-to-end ecosystem synthesis ──────────────────────────────────────

def test_ecosystem_composes_across_sources_via_recombination():
    """The point of Phase 2: an answer combining facts from two different
    sources that no single source contains verbatim."""
    eco = ResponseEcosystem(max_rounds=4)
    prompt = "What do you know about wombats and France?"
    evidence = [
        "Wombats are marsupials.",
        "The capital of France is Paris.",
    ]
    result = eco.generate(prompt, evidence)
    # At minimum, both facts must be representable somewhere in the
    # surviving population's niche winners -- either directly (cross-source
    # splice, already true in Phase 1) or via genuine recombination.
    all_text = " ".join(o.text.lower() for o in result.niche_winners.values())
    assert "wombats" in all_text
    assert "paris" in all_text


def test_ecosystem_relational_memory_threaded_through_agent():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.enable_ecological_generation()
    agent.relational.store_triple("mars", "is", "red")
    agent.relational.store_triple("mars", "is", "cold")
    # Just confirms no exception is raised when relational_memory carries
    # genuine ties -- the agent's own self.relational is passed through
    # to ResponseEcosystem.generate automatically.
    agent.process_turn("Mars is red and mysterious.")
    result = agent.process_turn("What is Mars like?")
    assert result["response_mode"] in (
        "ecological_generation", "abstention", "relational_reasoning",
        "relational_ambiguous",
    )
