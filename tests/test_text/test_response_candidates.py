from src.text import (
    FixedSpliceCandidateGenerator,
    LearnedChunkComposer,
    SequenceCandidateScorer,
)


def test_fixed_splicer_produces_complete_and_cross_source_candidates():
    generator = FixedSpliceCandidateGenerator()
    candidates = generator.generate(
        "How do I bake bread?",
        [
            "Mix flour and water. Let it rise.",
            "Preheat the oven. Bake until golden.",
        ],
    )
    kinds = {candidate["kind"] for candidate in candidates}
    assert "complete" in kinds
    assert "single_source_splice" in kinds
    assert "cross_source_splice" in kinds
    assert len(candidates) > 2


def test_learned_path_is_one_candidate_not_the_only_answer():
    composer = LearnedChunkComposer()
    composer.learn("bread", "Mix ingredients. Bake them.")
    candidates = FixedSpliceCandidateGenerator().generate(
        "bread", ["Use flour and water."], composer
    )
    assert any(item["kind"] == "learned_composition" for item in candidates)
    assert any(item["kind"] == "complete" for item in candidates)


def test_global_scorer_prefers_coherent_supported_complete_response():
    scorer = SequenceCandidateScorer()
    evidence = ["Mix flour and water. Let the dough rise."]
    candidates = [
        {
            "text": evidence[0], "kind": "complete", "source_ids": [0],
        },
        {
            "text": "Mix flour. Football teams score goals.",
            "kind": "cross_source_splice", "source_ids": [0, 1],
        },
    ]
    ranked = scorer.rank("How do I make dough?", candidates, evidence)
    assert ranked[0]["kind"] == "complete"


def test_pairwise_preference_updates_scorer():
    scorer = SequenceCandidateScorer()
    preferred = {"text": "Relevant answer.", "source_ids": [0]}
    rejected = {"text": "Unrelated repeated repeated.", "source_ids": [0, 1]}
    before = list(scorer.weights)
    scorer.learn_preference(
        "relevant question", preferred, rejected, ["Relevant answer."]
    )
    assert scorer.weights != before
    restored = SequenceCandidateScorer.from_state(scorer.get_state())
    assert restored.weights == scorer.weights


def test_repeated_preferences_change_candidate_ranking():
    scorer = SequenceCandidateScorer(learning_rate=0.2)
    short = {
        "text": "Brief answer.", "kind": "complete", "source_ids": [0],
    }
    detailed = {
        "text": "A relevant detailed answer with useful supporting context.",
        "kind": "human_preferred", "source_ids": [],
    }
    evidence = ["Brief answer."]
    before = scorer.score("relevant question", detailed, evidence)
    for _ in range(20):
        scorer.learn_preference(
            "relevant question", detailed, short, evidence
        )
    after = scorer.score("relevant question", detailed, evidence)
    assert after > before
    assert scorer.updates == 20
