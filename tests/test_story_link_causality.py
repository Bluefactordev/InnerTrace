"""
Test Story Link Causality - Verifies that story.link correctly reconstructs
context without duplication and with deterministic ordering.

v0.1: Critical test for validating the projection layer architecture.
"""
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from tracing import Tracer
from utils.storytelling.story_projection import StoryProjector


def test_story_link_causality():
    """
    BLOCKER TEST: Verifies that story.link → span_id reconstruction works correctly.

    Tests:
    1. llm.call.start/end events are emitted with prompt/response in blob store
    2. story.link event references llm.call via span_id
    3. StoryProjector correctly retrieves prompt/response from llm.call (no duplication)
    4. Ordering is deterministic
    5. No content is duplicated in story.link events
    """
    # Create temporary directory for test traces
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        events_path = tmpdir_path / "events.jsonl"
        blobs_path = tmpdir_path / "blobs"

        # Initialize tracer
        tracer = Tracer(
            events_path=str(events_path),
            blobs_path=str(blobs_path)
        )

        # Start a run
        run_id = tracer.start_run(
            entrypoint="test.story_link",
            args={"conversation_id": "test-conv-001"}
        )

        # Create a span for meta_planning phase
        with tracer.span(name="meta_planning", actor="meta_planner", kind="agent") as phase_span:
            # Emit llm.call.start
            llm_prompt = "Create a strategic plan for document analysis"
            llm_response = "Strategic objectives: 1) Extract metadata 2) Analyze content 3) Generate summary"

            # Store in blob store
            prompt_ref = tracer.blob_store.put(llm_prompt, ext="txt")
            response_ref = tracer.blob_store.put(llm_response, ext="txt")

            # Create a nested span for the LLM call
            with tracer.span(name="llm_call", actor="meta_planner", kind="llm") as llm_span_id:
                # Emit llm.call.start (with conversation_id for filtering)
                tracer.emit(
                    type="llm.call.start",
                    actor="meta_planner",
                    tags=["llm", "planning"],
                    payload={
                        "conversation_id": "test-conv-001",  # CRITICAL: needed for projector filtering
                        "model_id": "gpt-4",
                        "prompt_ref": prompt_ref,
                        "params": {"temperature": 0.7},
                        "purpose": "strategic_planning"
                    }
                )

                # Emit llm.call.end (same span, with conversation_id)
                tracer.emit(
                    type="llm.call.end",
                    actor="meta_planner",
                    tags=["llm", "planning"],
                    payload={
                        "conversation_id": "test-conv-001",  # CRITICAL: needed for projector filtering
                        "response_ref": response_ref,
                        "usage": {"input_tokens": 100, "output_tokens": 200},
                        "finish_reason": "stop",
                        "latency_ms": 1500
                    }
                )

                # NOW: Emit story.link (linking storytelling to execution)
                # This should NOT duplicate prompt/response, only reference via span_id
                tracer.emit(
                    type="story.link",
                    actor="story.manager",
                    tags=["storytelling", "llm_call", "meta_planning"],
                    payload={
                        "conversation_id": "test-conv-001",
                        "kind": "llm",
                        "target_span_id": llm_span_id,
                        "phase": "meta_planning",
                        "node": "meta_planner",
                        "model_id": "gpt-4",
                        "success": True
                    }
                )

        # End run
        tracer.end_run(status="ok")

        # === VERIFICATION PHASE ===

        # 1. Verify events were written to JSONL
        assert events_path.exists(), "events.jsonl should exist"

        # Load events
        events = []
        with open(events_path, 'r') as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

        print(f"✅ Total events emitted: {len(events)}")

        # Find specific events
        llm_start_events = [e for e in events if e.get("type") == "llm.call.start"]
        llm_end_events = [e for e in events if e.get("type") == "llm.call.end"]
        story_link_events = [e for e in events if e.get("type") == "story.link"]

        assert len(llm_start_events) == 1, f"Expected 1 llm.call.start, got {len(llm_start_events)}"
        assert len(llm_end_events) == 1, f"Expected 1 llm.call.end, got {len(llm_end_events)}"
        assert len(story_link_events) == 1, f"Expected 1 story.link, got {len(story_link_events)}"

        print("✅ Event counts correct")

        # 2. Verify story.link does NOT contain prompt/response (no duplication)
        story_link = story_link_events[0]
        story_payload = story_link.get("payload", {})

        assert "prompt_ref" not in story_payload, "story.link should NOT contain prompt_ref"
        assert "response_ref" not in story_payload, "story.link should NOT contain response_ref"
        assert "prompt" not in story_payload, "story.link should NOT contain prompt"
        assert "response" not in story_payload, "story.link should NOT contain response"

        print("✅ No content duplication in story.link")

        # 3. Verify story.link has target_span_id
        assert "target_span_id" in story_payload, "story.link should have target_span_id"
        target_span_id = story_payload["target_span_id"]
        assert target_span_id is not None, "target_span_id should not be None"

        print(f"✅ story.link has target_span_id: {target_span_id}")

        # 4. Verify llm.call events have the same span_id
        llm_start = llm_start_events[0]
        llm_end = llm_end_events[0]

        llm_start_span_id = llm_start.get("span_id")
        llm_end_span_id = llm_end.get("span_id")

        assert llm_start_span_id == llm_end_span_id, "llm.call.start and llm.call.end should have same span_id"
        assert llm_start_span_id == target_span_id, "story.link target_span_id should match llm.call span_id"

        print(f"✅ Span ID causality verified: {target_span_id}")

        # 5. Verify llm.call events have blob references
        llm_start_payload = llm_start.get("payload", {})
        llm_end_payload = llm_end.get("payload", {})

        assert "prompt_ref" in llm_start_payload, "llm.call.start should have prompt_ref"
        assert "response_ref" in llm_end_payload, "llm.call.end should have response_ref"

        print("✅ llm.call events have blob references")

        # DEBUG: Print what span IDs we have
        print(f"\nDEBUG: llm.call.start span_id: {llm_start_span_id}")
        print(f"DEBUG: llm.call.end span_id: {llm_end_span_id}")
        print(f"DEBUG: story.link target_span_id: {target_span_id}")

        # DEBUG: Print all span IDs in events
        all_span_ids = set()
        for e in events:
            if e.get("span_id"):
                all_span_ids.add(e.get("span_id"))
        print(f"DEBUG: All span_ids in events: {all_span_ids}")

        # 6. CRITICAL: Use StoryProjector to reconstruct
        projector = StoryProjector(
            events_path=str(events_path),
            blobs_path=str(blobs_path)
        )

        projection = projector.project("test-conv-001")

        assert projection is not None, "StoryProjector should reconstruct projection"
        print("✅ StoryProjector successfully reconstructed projection")

        # 7. Verify projection has LLM call in correct phase
        assert "meta_planning" in projection.phases, "Should have meta_planning phase"
        meta_phase = projection.phases["meta_planning"]

        assert len(meta_phase.llm_calls) == 1, f"Expected 1 LLM call in meta_planning, got {len(meta_phase.llm_calls)}"
        print("✅ Projection has 1 LLM call in meta_planning phase")

        # 8. CRITICAL: Verify prompt/response were retrieved from llm.call (not story.link)
        llm_call_semantic = meta_phase.llm_calls[0]

        assert llm_call_semantic.prompt == llm_prompt, "Prompt should match original (retrieved from blob)"
        assert llm_call_semantic.response == llm_response, "Response should match original (retrieved from blob)"
        assert llm_call_semantic.phase == "meta_planning", "Phase should be meta_planning"
        assert llm_call_semantic.node == "meta_planner", "Node should be meta_planner"
        assert llm_call_semantic.model_id == "gpt-4", "Model should be gpt-4"

        print("✅ Prompt/response correctly retrieved from llm.call events via span_id link")

        # 9. Verify deterministic ordering (events in temporal order)
        timestamps = [e.get("ts_iso") for e in events if e.get("ts_iso")]
        sorted_timestamps = sorted(timestamps)
        assert timestamps == sorted_timestamps, "Events should be in chronological order"

        print("✅ Deterministic ordering verified")

        # 10. Verify no data loss (all information preserved)
        assert llm_call_semantic.success == True, "Success flag should be preserved"
        assert llm_call_semantic.error is None, "Error should be None"

        print("✅ All data preserved without loss")

        print("\n" + "="*60)
        print("🎉 ALL CAUSALITY TESTS PASSED")
        print("="*60)
        print("\nVerified:")
        print("  ✅ story.link does NOT duplicate content")
        print("  ✅ story.link references llm.call via span_id")
        print("  ✅ StoryProjector correctly reconstructs from events")
        print("  ✅ Prompt/response retrieved from llm.call (not story.link)")
        print("  ✅ Ordering is deterministic")
        print("  ✅ No data loss in projection")
        print("\nv0.1 projection architecture: VALIDATED ✅")


if __name__ == "__main__":
    test_story_link_causality()
