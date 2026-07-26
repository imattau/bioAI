from experiments.library_size_sweep import (
    CHECKPOINTS,
    insertion_schedule,
    question_text,
    target_fact,
)


def test_schedule_adds_equal_answer_coverage_per_band():
    schedule = insertion_schedule()
    assert len(schedule) == len(CHECKPOINTS) * 20
    cumulative = 0
    previous = 0
    for checkpoint in CHECKPOINTS:
        cumulative += sum(previous < rank <= checkpoint for rank in schedule)
        assert cumulative == (CHECKPOINTS.index(checkpoint) + 1) * 20
        previous = checkpoint


def test_questions_are_paraphrases_not_exact_target_text():
    assert question_text(7) != target_fact(7)
    assert "regioncode7" in question_text(7)
    assert "regioncode7" in target_fact(7)
