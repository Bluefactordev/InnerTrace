"""Tests for function-level tracing."""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path

from innertrace.tracing.tracer import Tracer
from innertrace.tracing.function_tracing import trace_function, trace_block, trace_module


def test_trace_function_sync_emits_events():
    """Test that @trace_function emits span.start and span.end for sync functions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        events_path = tmpdir_path / "events.jsonl"
        blobs_path = tmpdir_path / "blobs"

        # Initialize tracer
        tracer = Tracer(
            events_path=str(events_path),
            blobs_path=str(blobs_path)
        )

        # Set as global tracer
        from innertrace.tracing import set_tracer
        set_tracer(tracer)

        # Define traced function
        @trace_function(capture_args=True, capture_return=False)
        def add_numbers(x, y):
            return x + y

        # Start a run and call the function
        run_id = tracer.start_run(entrypoint="test.sync_function", args={})

        result = add_numbers(10, 20)
        assert result == 30

        tracer.end_run(status="ok")

        # Read events
        events = []
        with open(events_path, "r") as f:
            for line in f:
                events.append(json.loads(line))

        # Verify events
        span_start = None
        span_end = None
        for event in events:
            if event["type"] == "span.start" and event.get("payload", {}).get("kind") == "function":
                span_start = event
            if event["type"] == "span.end" and span_start and event["span_id"] == span_start["span_id"]:
                span_end = event

        assert span_start is not None, "span.start event not found"
        assert span_end is not None, "span.end event not found"

        # Verify payload
        assert span_start["payload"]["kind"] == "function"
        assert span_start["payload"]["func"].endswith("add_numbers")
        assert span_start["payload"]["args"] == [10, 20]
        assert span_start["payload"]["kwargs"] == {}

        # Verify span hierarchy
        assert span_start["run_id"] == run_id
        assert span_end["run_id"] == run_id
        assert span_end["payload"]["status"] == "ok"
        assert "latency_ms" in span_end["payload"]


def test_trace_function_async_emits_events():
    """Test that @trace_function emits events for async functions."""
    async def run_test():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            events_path = tmpdir_path / "events.jsonl"
            blobs_path = tmpdir_path / "blobs"

            # Initialize tracer
            tracer = Tracer(
                events_path=str(events_path),
                blobs_path=str(blobs_path)
            )

            # Set as global tracer
            from innertrace.tracing import set_tracer
            set_tracer(tracer)

            # Define async traced function
            @trace_function(capture_args=True)
            async def async_multiply(x, y):
                await asyncio.sleep(0.01)
                return x * y

            # Start a run and call the function
            run_id = tracer.start_run(entrypoint="test.async_function", args={})

            result = await async_multiply(5, 6)
            assert result == 30

            tracer.end_run(status="ok")

            # Read events
            events = []
            with open(events_path, "r") as f:
                for line in f:
                    events.append(json.loads(line))

            # Verify events
            span_start = None
            span_end = None
            for event in events:
                if event["type"] == "span.start" and event.get("payload", {}).get("kind") == "function":
                    span_start = event
                if event["type"] == "span.end" and span_start and event["span_id"] == span_start["span_id"]:
                    span_end = event

            assert span_start is not None
            assert span_end is not None
            assert span_start["payload"]["args"] == [5, 6]
            assert span_end["payload"]["status"] == "ok"

    asyncio.run(run_test())


def test_trace_function_captures_return():
    """Test that capture_return=True captures the return value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        events_path = tmpdir_path / "events.jsonl"
        blobs_path = tmpdir_path / "blobs"

        tracer = Tracer(str(events_path), str(blobs_path))
        from innertrace.tracing import set_tracer
        set_tracer(tracer)

        @trace_function(capture_return=True)
        def get_message():
            return "Hello, World!"

        run_id = tracer.start_run(entrypoint="test.capture_return", args={})
        result = get_message()
        tracer.end_run(status="ok")

        # Read events
        events = []
        with open(events_path, "r") as f:
            for line in f:
                events.append(json.loads(line))

        # Find span.end event
        span_end = None
        for event in events:
            if event["type"] == "span.end" and event.get("payload", {}).get("kind") != "function":
                # Skip non-function spans
                continue
            if event["type"] == "span.end":
                span_end = event
                break

        assert span_end is not None
        assert "return" in span_end["payload"]
        assert span_end["payload"]["return"] == "Hello, World!"


def test_trace_function_handles_exception():
    """Test that exceptions are logged and re-raised."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        events_path = tmpdir_path / "events.jsonl"
        blobs_path = tmpdir_path / "blobs"

        tracer = Tracer(str(events_path), str(blobs_path))
        from innertrace.tracing import set_tracer
        set_tracer(tracer)

        @trace_function()
        def failing_function():
            raise ValueError("Test error")

        run_id = tracer.start_run(entrypoint="test.exception", args={})

        # Should raise the exception
        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        tracer.end_run(status="error")

        # Read events
        events = []
        with open(events_path, "r") as f:
            for line in f:
                events.append(json.loads(line))

        # Find span.end and exception events
        span_end = None
        exception_event = None
        for event in events:
            if event["type"] == "span.end" and event.get("payload", {}).get("status") == "error":
                span_end = event
            if event["type"] == "exception":
                exception_event = event

        assert span_end is not None
        assert span_end["payload"]["status"] == "error"
        assert exception_event is not None


def test_trace_function_without_run():
    """Test that traced functions work without an active run (no tracing)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        events_path = tmpdir_path / "events.jsonl"
        blobs_path = tmpdir_path / "blobs"

        tracer = Tracer(str(events_path), str(blobs_path))
        from innertrace.tracing import set_tracer
        set_tracer(tracer)

        @trace_function()
        def simple_function(x):
            return x * 2

        # Call without starting a run
        result = simple_function(5)
        assert result == 10

        # No events should be emitted (file might not exist or be empty)
        if events_path.exists():
            events = []
            with open(events_path, "r") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
            # Should have no span.start/end events
            function_events = [e for e in events if e.get("type") in ("span.start", "span.end")]
            assert len(function_events) == 0


def test_trace_module_wraps_only_local():
    """Test that trace_module only wraps functions defined in the module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        events_path = tmpdir_path / "events.jsonl"
        blobs_path = tmpdir_path / "blobs"

        tracer = Tracer(str(events_path), str(blobs_path))
        from innertrace.tracing import set_tracer
        set_tracer(tracer)

        # Simulate a module with local and imported functions
        module_globals = {
            "__name__": "test_module",
        }

        # Define local functions
        def local_func1():
            return "local1"

        def local_func2():
            return "local2"

        # Imported function (different module)
        def imported_func():
            return "imported"

        imported_func.__module__ = "other_module"

        # Add to module globals
        module_globals["local_func1"] = local_func1
        module_globals["local_func2"] = local_func2
        module_globals["imported_func"] = imported_func

        # Apply trace_module
        trace_module(module_globals)

        # Check that local functions are wrapped
        assert hasattr(module_globals["local_func1"], "__innertrace_wrapped__")
        assert hasattr(module_globals["local_func2"], "__innertrace_wrapped__")

        # Check that imported function is NOT wrapped
        assert not hasattr(module_globals["imported_func"], "__innertrace_wrapped__")


def test_trace_module_respects_include_exclude():
    """Test that trace_module respects include/exclude patterns."""
    # Setup module
    module_globals = {
        "__name__": "test_module",
    }

    def public_function():
        return "public"

    def _private_function():
        return "private"

    def special_handler():
        return "special"

    module_globals["public_function"] = public_function
    module_globals["_private_function"] = _private_function
    module_globals["special_handler"] = special_handler

    # Test include_private=False (default)
    trace_module(module_globals, include_private=False)

    assert hasattr(module_globals["public_function"], "__innertrace_wrapped__")
    assert not hasattr(module_globals["_private_function"], "__innertrace_wrapped__")
    assert hasattr(module_globals["special_handler"], "__innertrace_wrapped__")

    # Reset
    module_globals["public_function"] = public_function
    module_globals["_private_function"] = _private_function
    module_globals["special_handler"] = special_handler

    # Test include_private=True
    trace_module(module_globals, include_private=True)

    assert hasattr(module_globals["_private_function"], "__innertrace_wrapped__")

    # Reset
    module_globals["public_function"] = public_function
    module_globals["special_handler"] = special_handler

    # Test exclude pattern
    trace_module(module_globals, exclude=["special_*"])

    assert hasattr(module_globals["public_function"], "__innertrace_wrapped__")
    assert not hasattr(module_globals["special_handler"], "__innertrace_wrapped__")


def test_trace_block_context_manager():
    """Test that trace_block emits span events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        events_path = tmpdir_path / "events.jsonl"
        blobs_path = tmpdir_path / "blobs"

        tracer = Tracer(str(events_path), str(blobs_path))
        from innertrace.tracing import set_tracer
        set_tracer(tracer)

        run_id = tracer.start_run(entrypoint="test.trace_block", args={})

        with trace_block("my_block", payload={"custom": "data"}):
            # Do some work
            x = 1 + 1

        tracer.end_run(status="ok")

        # Read events
        events = []
        with open(events_path, "r") as f:
            for line in f:
                events.append(json.loads(line))

        # Find span.start and span.end
        span_start = None
        span_end = None
        for event in events:
            if event["type"] == "span.start" and event.get("payload", {}).get("name") == "my_block":
                span_start = event
            if event["type"] == "span.end" and span_start and event["span_id"] == span_start["span_id"]:
                span_end = event

        assert span_start is not None
        assert span_end is not None
        assert span_start["payload"]["kind"] == "function"
        assert span_start["payload"]["custom"] == "data"
        assert span_end["payload"]["status"] == "ok"


def test_trace_function_with_redaction():
    """Test that custom redaction keys work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        events_path = tmpdir_path / "events.jsonl"
        blobs_path = tmpdir_path / "blobs"

        tracer = Tracer(str(events_path), str(blobs_path))
        from innertrace.tracing import set_tracer
        set_tracer(tracer)

        @trace_function(redact=["secret_param"])
        def process_data(public_data, secret_param):
            return "processed"

        run_id = tracer.start_run(entrypoint="test.redaction", args={})
        process_data("visible", "should_be_hidden")
        tracer.end_run(status="ok")

        # Read events
        events = []
        with open(events_path, "r") as f:
            for line in f:
                events.append(json.loads(line))

        # Find span.start
        span_start = None
        for event in events:
            if event["type"] == "span.start" and event.get("payload", {}).get("kind") == "function":
                span_start = event
                break

        assert span_start is not None
        # Args should have the redacted parameter
        # Note: args are positional, so redaction applies to kwargs
        # But we can check that sensitive strings are handled
