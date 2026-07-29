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
from src.text.ecology.frame_extractor import FrameLibrary
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


# ── Phase 7: compound-predicate extraction ──────────────────────────────

def test_extract_propositions_splits_compound_predicate():
    """Regression test for a real bug found via Phase 7's frozen-spec
    check: PropositionRealiser's compound-clause merging can produce
    "Cows were farm animals and sit in farmyard and have long horns." --
    three relations sharing one subject. ConsolidationMemory.extract_relations's
    catch-all pattern still *matches* this (any "was" followed by
    anything), garbling everything after "were" into one bogus object
    unless the compound shape is detected and split first."""
    memories = ["Cows were farm animals and sit in farmyard and have long horns."]
    props = extract_propositions(memories)
    triples = {(p.subject, p.relation, p.object) for p in props}
    assert triples == {
        ("cows", "is", "farm animals"),
        ("cows", "in", "farmyard"),
        ("cows", "has", "long horns"),
    }


def test_extract_propositions_two_way_compound():
    memories = ["Wombats are marsupials and are in australia."]
    props = extract_propositions(memories)
    triples = {(p.subject, p.relation, p.object) for p in props}
    assert triples == {
        ("wombats", "is", "marsupials"),
        ("wombats", "in", "australia"),
    }


def test_extract_propositions_does_not_split_legitimate_and_joined_object():
    """"marsupials and mammals" is one relation's multi-value object (a
    pre-existing, Phase 2 behavior) -- "mammals" doesn't start with a
    verb, so this must NOT be treated as a compound predicate."""
    memories = ["Wombats are marsupials and mammals."]
    props = extract_propositions(memories)
    assert [(p.subject, p.relation, p.object) for p in props] == [
        ("wombats", "is", "marsupials and mammals")
    ]


# ── Phase 9: pluggable extraction sources (fixed / learned / hybrid) ────

def test_extract_propositions_default_flags_match_original_behavior():
    """allow_fixed_patterns=True, allow_learned_frames=False is the
    default -- every pre-Phase-9 call site is unaffected."""
    memories = ["Wombats are marsupials."]
    assert extract_propositions(memories) == extract_propositions(
        memories, frame_library=FrameLibrary(),
        allow_fixed_patterns=True, allow_learned_frames=False,
    )


def test_extract_propositions_fixed_only_ignores_unparseable_construction():
    memories = ["Wombats belong to the marsupial family."]
    assert extract_propositions(memories) == []


def test_extract_propositions_learned_frames_recover_unparseable_construction():
    library = FrameLibrary()
    library.observe_labelled(
        "Koalas belong to the eucalyptus family.", "koalas", "is", "eucalyptus",
    )
    memories = ["Wombats belong to the marsupial family."]
    props = extract_propositions(
        memories, frame_library=library,
        allow_fixed_patterns=False, allow_learned_frames=True,
    )
    assert [(p.subject, p.relation, p.object) for p in props] == [
        ("wombats", "is", "marsupial")
    ]


def test_extract_propositions_hybrid_prefers_fixed_pattern_when_both_match():
    """Hybrid tries the fixed regex first per sentence, falling back to
    learned frames only when the fixed pass finds nothing -- confirmed by
    a sentence both sources could plausibly claim."""
    library = FrameLibrary()
    library.observe_labelled("Wombats are marsupials.", "wombats", "is", "marsupials")
    memories = ["Wombats are marsupials."]
    props = extract_propositions(
        memories, frame_library=library,
        allow_fixed_patterns=True, allow_learned_frames=True,
    )
    # Exactly one triple -- not duplicated by also matching via the
    # learned frame, since the fixed pass already found something.
    assert [(p.subject, p.relation, p.object) for p in props] == [
        ("wombats", "is", "marsupials")
    ]


def test_extract_propositions_hybrid_falls_back_to_learned_frame():
    library = FrameLibrary()
    library.observe_labelled(
        "Koalas belong to the eucalyptus family.", "koalas", "is", "eucalyptus",
    )
    memories = ["Wombats belong to the marsupial family."]
    props = extract_propositions(
        memories, frame_library=library,
        allow_fixed_patterns=True, allow_learned_frames=True,
    )
    assert [(p.subject, p.relation, p.object) for p in props] == [
        ("wombats", "is", "marsupial")
    ]


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
    # Plural subject and noun objects, not "mars"/adjectives: "mars" is a
    # real, disclosed is_plural_noun limitation (lemminflect has no
    # dictionary entry for the proper noun and falls back to treating the
    # trailing "s" as a regular plural suffix -- see realiser.py's
    # _AGREEMENT_EXEMPT_RELATIONS comment for the same issue on
    # "capital_of"), and "is" is scoped to noun objects for Phase 7's
    # article insertion, not adjectives -- this test is about the
    # pre-existing (Phase 2) same-(subject,relation) "and"-joining, not
    # either of those, so it uses content that doesn't trip either edge.
    props = (
        Proposition("wombats", "is", "marsupials", 0, "x"),
        Proposition("wombats", "is", "mammals", 0, "y"),
    )
    text = PropositionRealiser.realise(props)
    assert "marsupials and mammals" in text.lower()


def test_realiser_empty_propositions_gives_empty_string():
    assert PropositionRealiser.realise(()) == ""


def test_realiser_capitalizes_every_clause_not_just_the_first():
    # Two DIFFERENT subjects, not one subject across two relations --
    # since Phase 7, the latter now correctly compound-merges into one
    # sentence ("Wombats are marsupials and are in Australia."), which no
    # longer exercises the multi-*sentence* capitalization this test is
    # actually about.
    props = (
        Proposition("wombats", "is", "marsupials", 0, "x"),
        Proposition("elephants", "is", "herbivores", 1, "y"),
    )
    text = PropositionRealiser.realise(props)
    assert ". Elephants" in text
    assert ". elephants" not in text


# ── Phase 7: compound-clause merging and article insertion ─────────────

def test_realiser_merges_different_relations_for_same_subject():
    props = (
        Proposition("wombats", "is", "marsupials", 0, "x"),
        Proposition("wombats", "in", "australia", 1, "y"),
    )
    text = PropositionRealiser.realise(props)
    assert text == "Wombats are marsupials and are in australia."
    assert ". " not in text


def test_realiser_merges_three_relations_for_same_subject():
    props = (
        Proposition("wombats", "is", "marsupials", 0, "x"),
        Proposition("wombats", "in", "australia", 1, "y"),
        Proposition("wombats", "has", "strong claws", 2, "z"),
    )
    text = PropositionRealiser.realise(props)
    assert text == "Wombats are marsupials and are in australia and have strong claws."


def test_realiser_does_not_merge_capital_relation_into_other_clauses():
    """"capital"'s template doesn't start with [SUBJECT] (its real
    grammatical subject is "the capital", not the country) -- it must
    never be folded into a compound clause with another relation about
    the same subject, unlike is/in/has."""
    props = (
        Proposition("france", "is", "country", 0, "x"),
        Proposition("france", "capital", "paris", 1, "y"),
    )
    text = PropositionRealiser.realise(props)
    assert " and the capital of" not in text.lower()
    assert text.count(".") == 2  # two separate sentences


def test_realiser_compound_clause_still_traceable_to_propositions():
    props = (
        Proposition("wombats", "is", "marsupials", 0, "x"),
        Proposition("wombats", "in", "australia", 1, "y"),
    )
    text = PropositionRealiser.realise(props).lower()
    for prop in props:
        assert prop.subject in text
        assert prop.object in text


def test_realiser_capital_of_subject_never_agreed_as_plural():
    """Regression test for a real bug found via Phase 7 testing:
    is_plural_noun("paris") returns True (lemminflect has no dictionary
    entry for the proper noun and infers a nonexistent singular "pari"
    from the trailing "s"). "capital_of"'s subject is always a city --
    grammatically singular by definition -- so it must be exempted from
    agreement entirely rather than trusting surface morphology."""
    props = (Proposition("paris", "capital_of", "france", 0, "x"),)
    text = PropositionRealiser.realise(props)
    assert text == "Paris is the capital of france."


def test_realiser_adds_article_for_singular_is_object():
    props = (Proposition("wombat", "is", "marsupial", 0, "x"),)
    assert PropositionRealiser.realise(props) == "Wombat is a marsupial."


def test_realiser_adds_article_for_singular_has_object():
    props = (Proposition("wombat", "has", "strong claw", 0, "x"),)
    assert PropositionRealiser.realise(props) == "Wombat has a strong claw."


def test_realiser_no_article_for_plural_is_object():
    props = (Proposition("wombats", "is", "marsupials", 0, "x"),)
    assert PropositionRealiser.realise(props) == "Wombats are marsupials."


def test_realiser_no_article_for_in_or_capital_relations():
    """Article insertion is deliberately scoped to is/has only -- in/
    capital/capital_of objects are reliably place/city/country names
    (proper nouns) in this curriculum, which never take an article."""
    props = (Proposition("wombat", "in", "australia", 0, "x"),)
    text = PropositionRealiser.realise(props)
    assert " a australia" not in text.lower()
    assert " an australia" not in text.lower()


# ── Phase 8: possessive pronoun for "capital" ───────────────────────────

def test_realiser_keeps_full_noun_phrase_when_capital_has_no_prior_mention():
    """No earlier compound clause introduced "france" -- a pronoun here
    would have no antecedent, so "capital" keeps its full noun-phrase
    template exactly as before Phase 8."""
    props = (Proposition("france", "capital", "paris", 0, "x"),)
    text = PropositionRealiser.realise(props)
    assert text == "The capital of france is paris."


def test_realiser_uses_possessive_pronoun_for_capital_after_prior_clause():
    """"france" was already introduced by an earlier is/in/has compound
    clause for the same subject -- "capital" now refers back to it with a
    possessive pronoun instead of repeating the full noun phrase."""
    props = (
        Proposition("france", "is", "country", 0, "x"),
        Proposition("france", "capital", "paris", 1, "y"),
    )
    text = PropositionRealiser.realise(props)
    assert text == "France is a country. Its capital is paris."
    assert "the capital of" not in text.lower()


def test_realiser_uses_plural_possessive_pronoun_for_capital():
    props = (
        Proposition("islands", "is", "country", 0, "x"),
        Proposition("islands", "capital", "paris", 1, "y"),
    )
    text = PropositionRealiser.realise(props)
    assert text == "Islands are a country. Their capital is paris."


def test_realiser_country_is_in_clauses_still_compound_merge():
    """Phase 8 only changes "capital"'s rendering -- is/in for a country
    subject still merge into one compound clause exactly as any other
    subject's is/in/has propositions do (this phase doesn't touch that
    path)."""
    props = (
        Proposition("france", "is", "country", 0, "x"),
        Proposition("france", "in", "western europe", 1, "y"),
    )
    text = PropositionRealiser.realise(props)
    assert text == "France is a country and is in western europe."


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


def test_ecosystem_final_winner_prefers_supported_proposition_count():
    """Regression test for a real bug found while building Phase 3's
    benchmark: comparing raw niche_score across niches with different
    weight vectors is apples-to-oranges, so the short "direct" niche's
    single-fact answer almost always won the overall comparison even when
    an "integrative"/"explanatory" niche_winner correctly composed both
    known facts. The final `winner`/`response` must prefer whichever
    survivor's realised text actually asserts more of the evidence's known
    propositions, not just whichever niche happened to score highest on
    its own internal scale."""
    eco = ResponseEcosystem(max_rounds=4)
    prompt = "What do you know about wombats?"
    evidence = ["Wombats are marsupials.", "Wombats are in Australia."]
    result = eco.generate(prompt, evidence)
    text = result.response.lower()
    assert "marsupials" in text
    assert "australia" in text


def test_ecosystem_recombines_even_when_genotype_organisms_lose_their_niche_slot():
    """Regression test for a real bug found while building Phase 3's
    benchmark: two short, single-proposition organisms about the same
    subject (one per source) can both lose their niche slot to already-
    good raw candidates in the same niche before ever getting a chance to
    recombine -- silently defeating Phase 2's whole point for exactly the
    short, one-fact-per-source case it's meant to help with. Genotype
    organisms must get to participate in reproduction regardless of
    whether they won a niche slot that round."""
    eco = ResponseEcosystem(max_rounds=4, survivors_per_niche=2)
    prompt = "What do you know about wombats?"
    evidence = ["Wombats are marsupials.", "Wombats are in Australia."]
    result = eco.generate(prompt, evidence)
    multi_proposition_winner = any(
        len(organism.propositions) >= 2
        for organism in result.niche_winners.values()
    )
    assert multi_proposition_winner


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


# ── Phase 4: frame-aware realisation ────────────────────────────────────

def test_realiser_uses_observed_frame_when_given_a_library():
    library = FrameLibrary()
    library.observe("The cat sits within the box.")  # teaches an "in" frame
    props = (Proposition("wombats", "in", "australia", 0, "x"),)

    without_library = PropositionRealiser.realise(props)
    with_library = PropositionRealiser.realise(props, frame_library=library)

    # "sit within", not "sits within" -- Phase 5's subject-verb agreement
    # correctly re-inflects the verb for the plural subject "wombats",
    # even though the frame was originally observed with a singular one.
    assert "within" in with_library.lower()
    assert "sit within" in with_library.lower()
    assert "within" not in without_library.lower()


def test_realiser_falls_back_to_fixed_template_when_library_has_nothing():
    library = FrameLibrary()  # empty -- nothing observed for "in"
    props = (Proposition("wombats", "in", "australia", 0, "x"),)
    assert PropositionRealiser.realise(
        props, frame_library=library
    ) == PropositionRealiser.realise(props)


def test_realiser_require_frame_returns_empty_when_no_frame_available():
    library = FrameLibrary()  # empty -- nothing observed for "in"
    props = (Proposition("wombats", "in", "australia", 0, "x"),)
    assert PropositionRealiser.realise(
        props, frame_library=library, require_frame=True
    ) == ""


def test_realiser_require_frame_returns_empty_with_no_frame_library_at_all():
    props = (Proposition("wombats", "in", "australia", 0, "x"),)
    assert PropositionRealiser.realise(props, require_frame=True) == ""


def test_realiser_require_frame_succeeds_when_every_relation_has_a_frame():
    library = FrameLibrary()
    library.observe("The cat sits within the box.")
    props = (Proposition("wombats", "in", "australia", 0, "x"),)
    text = PropositionRealiser.realise(
        props, frame_library=library, require_frame=True
    )
    assert text != ""
    assert "sit within" in text.lower()


def test_realiser_require_frame_does_not_block_possessive_pronoun_branch():
    """The capital-possessive branch doesn't consult frame_library at all
    -- require_frame must not treat it as a missing-frame failure."""
    library = FrameLibrary()  # empty
    props = (
        Proposition("france", "is", "country", 0, "x"),
        Proposition("france", "capital", "paris", 1, "y"),
    )
    text = PropositionRealiser.realise(
        props, frame_library=library, require_frame=True
    )
    assert text == ""  # "is" has no frame either -- whole call fails
    # But once "is" has a frame too, the possessive branch must not be
    # the reason for failure.
    library.observe("Spain is a country.")
    text2 = PropositionRealiser.realise(
        props, frame_library=library, require_frame=True
    )
    assert text2 != ""
    assert "its capital is paris" in text2.lower()


def test_ecosystem_frame_library_influences_realised_output():
    library = FrameLibrary()
    library.observe("The cat sits within the box.")
    eco = ResponseEcosystem(max_rounds=4)
    prompt = "What do you know about wombats?"
    evidence = ["Wombats are marsupials.", "Wombats are in Australia."]
    result = eco.generate(prompt, evidence, frame_library=library)
    # Checked across niche_winners, not the specific final response --
    # which niche wins the overall cross-niche comparison isn't what this
    # test is about (see test_ecosystem_final_winner_prefers_supported_proposition_count
    # for that); this just confirms the frame-derived phrasing genuinely
    # gets produced and survives into the population somewhere.
    all_text = " ".join(o.text.lower() for o in result.niche_winners.values())
    assert "within" in all_text


def test_ecosystem_evidence_propositions_bypasses_text_extraction():
    """Phase 9: evidence_propositions lets a caller hand the ecosystem
    structured facts directly, even when the evidence TEXT is unparseable
    gibberish -- isolating generation quality from extraction accuracy."""
    eco = ResponseEcosystem(max_rounds=4)
    prompt = "What do you know about wombats?"
    gibberish_evidence = ["wombat marsupial australia claws xyzzy plugh"]
    propositions = [
        Proposition("wombats", "is", "marsupials", 0, gibberish_evidence[0]),
        Proposition("wombats", "in", "australia", 0, gibberish_evidence[0]),
    ]
    result = eco.generate(
        prompt, gibberish_evidence, evidence_propositions=propositions,
    )
    # The final winner (picked by supported-proposition-count, Phase 3) is
    # a composed organism built from the supplied propositions -- not the
    # raw gibberish text, which still seeds a "direct" niche candidate
    # (unrelated to evidence_propositions, that's plain Phase 1 behavior)
    # but never wins the overall comparison since it asserts nothing.
    assert result.winner is not None
    assert "marsupials" in result.winner.text.lower()
    assert "australia" in result.winner.text.lower()


def test_agent_learn_conversation_populates_frame_library():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.learn_conversation("Tell me about the box.", "The cat sits within the box.")
    assert "[SUBJECT] sits within [OBJECT]" in agent.frame_library.frames


def test_agent_frame_library_persists_across_save_load():
    import tempfile
    from pathlib import Path

    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.learn_conversation("Tell me about the box.", "The cat sits within the box.")
    path = Path(tempfile.mktemp(suffix=".pt"))
    try:
        agent.save(path)
        restored = BioAIDialogueAgent.load(path)
        assert (
            "[SUBJECT] sits within [OBJECT]" in restored.frame_library.frames
        )
    finally:
        path.unlink(missing_ok=True)
