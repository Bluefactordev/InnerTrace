"""Regression tests for JSON-safe, redacted tool payload tracing."""

import json
from datetime import datetime, timezone

from innertrace.tracing.tracer import (
    Tracer,
    emit_llm_call_end,
    emit_tool_call_end,
    emit_tool_call_start,
)


def _read_events(events_path):
    return [json.loads(line) for line in events_path.read_text().splitlines()]


def test_tool_payloads_are_json_safe_and_redacted(tmp_path):
    events_path = tmp_path / "events.jsonl"
    tracer = Tracer(str(events_path), str(tmp_path / "blobs"))
    tracer.start_run("test.tool-payloads")

    emitted_at = datetime(2026, 8, 20, tzinfo=timezone.utc)
    emit_tool_call_start(
        tracer,
        "calendar.create",
        {"password": "do-not-store", "when": emitted_at, "coordinates": (1, 2)},
    )
    emit_tool_call_end(
        tracer,
        "calendar.create",
        {"token": "do-not-store", "created_at": emitted_at},
    )
    emit_llm_call_end(
        tracer,
        "done",
        tool_calls=[
            {
                "name": "calendar.create",
                "args": {"api_key": "do-not-store", "when": emitted_at},
            }
        ],
    )
    tracer.end_run()

    events = _read_events(events_path)
    start = next(event for event in events if event["type"] == "tool.call.start")
    end = next(event for event in events if event["type"] == "tool.call.end")
    llm_end = next(event for event in events if event["type"] == "llm.call.end")

    start_blob = json.loads(tracer.blob_store.get(start["payload"]["args_ref"]))
    end_blob = json.loads(tracer.blob_store.get(end["payload"]["result_ref"]))
    llm_args_blob = json.loads(
        tracer.blob_store.get(llm_end["payload"]["tool_calls"][0]["args_ref"])
    )

    assert start_blob == {
        "coordinates": [1, 2],
        "password": "***REDACTED***",
        "when": str(emitted_at),
    }
    assert end_blob == {
        "created_at": str(emitted_at),
        "token": "***REDACTED***",
    }
    assert llm_args_blob == {
        "api_key": "***REDACTED***",
        "when": str(emitted_at),
    }
    assert start["payload"]["args_preview"]["password"] == "***REDACTED***"
