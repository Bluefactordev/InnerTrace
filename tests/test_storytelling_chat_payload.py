"""Tests for the chat storytelling preview cursor."""

from innertrace.utils.storytelling.storytelling_extractor import (
    StorytellingExtractor,
    StorytellingSummary,
)


def _summary(total_events=7):
    return StorytellingSummary(
        conversation_id="conversation-1",
        created_at="2026-08-20T00:00:00Z",
        total_events=total_events,
        strategic_objectives=[],
        tactical_plans=[],
        completed_tasks=[],
        llm_calls=[],
        reflections=[],
        final_synthesis=None,
        duration_minutes=1.5,
        statistics={"completed_tasks": 0, "llm_calls": 0},
        task_quality_metrics=[],
        phase_quality_metrics=[],
        global_difficulty_score=0.0,
        avg_task_quality=0.0,
        execution_assessment="Unknown",
    )


def test_chat_payload_returns_preview_and_current_event_cursor(tmp_path, monkeypatch):
    extractor = StorytellingExtractor(
        storytelling_dir=str(tmp_path / "storytelling"),
        events_path=str(tmp_path / "events.jsonl"),
        blobs_path=str(tmp_path / "blobs"),
    )
    monkeypatch.setattr(extractor, "extract_summary", lambda _conversation_id: _summary())

    payload = extractor.get_chat_storytelling_payload("conversation-1")

    assert "conversation-1" in payload["storytelling"]
    assert payload["total_events"] == 7
    assert payload["last_event_index"] == 7


def test_chat_payload_returns_zero_cursor_when_storytelling_is_missing(tmp_path, monkeypatch):
    extractor = StorytellingExtractor(storytelling_dir=str(tmp_path / "storytelling"))
    monkeypatch.setattr(extractor, "extract_summary", lambda _conversation_id: None)

    payload = extractor.get_chat_storytelling_payload("missing")

    assert payload["total_events"] == 0
    assert payload["last_event_index"] == 0
    assert "non disponibile" in payload["storytelling"]
