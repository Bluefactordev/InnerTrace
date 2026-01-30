import json
import random
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

from innertrace import ReplayContext, NonDeterminismDetected, tool as replay_tool
from innertrace.replay import ReplayManager
from innertrace.tracing.blob_store import BlobStore


def _write_events(events_path: Path, events) -> None:
    with open(events_path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def _install_dummy_openai():
    module = types.ModuleType("openai")

    class Completions:
        def create(self, *args, **kwargs):
            raise AssertionError("Should be replayed")

    class AsyncCompletions:
        async def create(self, *args, **kwargs):
            raise AssertionError("Should be replayed")

    completions = types.SimpleNamespace(
        Completions=Completions,
        AsyncCompletions=AsyncCompletions,
    )
    chat = types.SimpleNamespace(completions=completions)
    module.resources = types.SimpleNamespace(chat=chat)

    class ChatCompletion:
        def create(self, *args, **kwargs):
            raise AssertionError("Should be replayed")

    module.ChatCompletion = ChatCompletion
    sys.modules["openai"] = module
    return module


def _prepare_replay_files(tmp_path: Path):
    run_id = "run-123"
    events_dir = tmp_path / "traces" / "events"
    blobs_dir = tmp_path / "traces" / "blobs"
    events_dir.mkdir(parents=True, exist_ok=True)

    blob_store = BlobStore(str(blobs_dir))
    prompt = [{"role": "user", "content": "hi"}]
    prompt_ref = blob_store.put(prompt, "json")
    response_ref = blob_store.put("hello", "txt")
    tool_result_ref = blob_store.put({"value": 42}, "json")
    args_ref = blob_store.put({"query": "abc"}, "json")

    events = [
        {
            "type": "run.start",
            "ts_iso": "2024-01-01T00:00:00",
            "payload": {"random_seed": 1234},
        },
        {"type": "llm.call.start", "payload": {"model": "gpt-4", "prompt_ref": prompt_ref}},
        {"type": "llm.call.end", "payload": {"response_ref": response_ref}},
        {"type": "tool.call.start", "payload": {"tool": "search", "args_ref": args_ref}},
        {
            "type": "tool.call.end",
            "payload": {"tool": "search", "status": "ok", "result_ref": tool_result_ref},
        },
    ]
    _write_events(events_dir / f"{run_id}.jsonl", events)
    return run_id, events_dir, blobs_dir


def test_replay_openai_and_tool(tmp_path):
    run_id, events_dir, blobs_dir = _prepare_replay_files(tmp_path)
    openai = _install_dummy_openai()
    call_counter = {"count": 0}

    @replay_tool(name="search")
    def search(query: str):
        call_counter["count"] += 1
        return {"value": 999}

    try:
        with ReplayContext(run_id, events_dir=str(events_dir), blobs_dir=str(blobs_dir)):
            response = openai.resources.chat.completions.Completions().create(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4",
            )
            assert response.choices[0].message.content == "hello"
            result = search(query="abc")
            assert result == {"value": 42}
            assert call_counter["count"] == 0
            assert datetime.now().isoformat().startswith("2024-01-01T00:00:00")
            assert random.random() == pytest.approx(0.9664535356921388)
    finally:
        sys.modules.pop("openai", None)


def test_replay_prompt_validation(tmp_path):
    run_id, events_dir, blobs_dir = _prepare_replay_files(tmp_path)
    manager = ReplayManager(
        run_id,
        events_dir=str(events_dir),
        blobs_dir=str(blobs_dir),
        validate_prompts=True,
    )
    with pytest.raises(NonDeterminismDetected):
        manager.next_llm_response([{"role": "user", "content": "different"}], model="gpt-4")
