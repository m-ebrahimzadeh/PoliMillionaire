from pathlib import Path

from polimibot import (
    GameSummaryRecord, QuestionRecord, RunLogger, load_jsonl,
)


def test_logger_writes_manifest_then_records(tmp_path: Path):
    with RunLogger(tmp_path, run_id="unit_test", extra={"model": "fake"}) as log:
        log.log_question(QuestionRecord(
            session_id=1, competition_id=2, competition_name="Sci",
            level=3, question_text="what?", options=["a","b","c","d"],
            chosen_index=1, chosen_text="b", correct=True, latency_seconds=0.5,
            strategy="baseline", confidence=0.7,
        ))
        log.log_summary(GameSummaryRecord(
            session_id=1, competition_id=2, competition_name="Sci",
            final_level=3, earned_amount=300.0, n_questions=1, n_correct=1,
        ))

    files = list(tmp_path.glob("run_*_unit_test.jsonl"))
    assert len(files) == 1

    records = list(load_jsonl(files[0]))
    assert len(records) == 3
    assert records[0]["run_kind"] == "manifest"
    assert records[0]["run_id"] == "unit_test"
    assert records[0]["extra"]["model"] == "fake"
    assert records[1]["run_kind"] == "question"
    assert records[1]["correct"] is True
    assert records[2]["run_kind"] == "summary"


def test_logger_outside_context_raises(tmp_path: Path):
    log = RunLogger(tmp_path, run_id="never_opened")
    import pytest
    with pytest.raises(RuntimeError):
        log.log_question(QuestionRecord())