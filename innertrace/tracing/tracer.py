"""Core Tracer implementation for LLM execution observability."""

import contextvars
import json
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from ulid import ULID
    def generate_id() -> str:
        return str(ULID())
except ImportError:
    import uuid
    def generate_id() -> str:
        return str(uuid.uuid4())

from .blob_store import BlobStore
from .redact import redact_payload, truncate_preview


# Event Type Vocabulary
# ======================
# The tracer uses a closed vocabulary of event types for consistency.
#
# Core execution events:
#   - run.start, run.end: Execution run lifecycle
#   - span.start, span.end: Hierarchical span tracking
#   - exception: Exception/error tracking
#
# LLM events:
#   - llm.call.start, llm.call.end: LLM API calls
#
# Tool events:
#   - tool.call.start, tool.call.end: Tool/function executions
#
# Router events:
#   - router.decision: Routing/branching decisions
#
# Sandbox events:
#   - sandbox.exec.start, sandbox.exec.end: Code execution in sandbox
#
# Storytelling events (semantic layer):
#
# v0.1 STABLE VOCABULARY (guaranteed stable across v0.1.x):
#   - story.phase.start, story.phase.end: Phase transitions (meta_planning, project_planning, execution, synthesis)
#   - story.objective: Strategic objective defined or updated
#   - story.task: Task event (planned, started, completed)
#   - story.link: Link storytelling context to execution event (via target_span_id)
#
# EXPERIMENTAL (may change in future versions):
#   - story.quality.task: Task quality metrics (quality_score, difficulty, insights)
#   - story.quality.phase: Phase quality metrics (success_rate, bottlenecks)
#
# Note: Storytelling events link to execution events via conversation_id in payload.
#       Large content (prompts, responses, descriptions) should use blob references.

# v0.1 Stable vocabulary (closed set for reliable parsing)
STORYTELLING_EVENT_TYPES_STABLE = [
    "story.phase.start",
    "story.phase.end",
    "story.objective",
    "story.task",
    "story.link",
]

# Experimental event types (may change or be removed in future versions)
STORYTELLING_EVENT_TYPES_EXPERIMENTAL = [
    "story.quality.task",
    "story.quality.phase",
]

# All storytelling event types (stable + experimental)
STORYTELLING_EVENT_TYPES = STORYTELLING_EVENT_TYPES_STABLE + STORYTELLING_EVENT_TYPES_EXPERIMENTAL


# Thread-local storage for span stack
_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('run_id', default=None)
_span_stack: contextvars.ContextVar[List[str]] = contextvars.ContextVar('span_stack', default=[])
_span_kind_stack: contextvars.ContextVar[List[str]] = contextvars.ContextVar('span_kind_stack', default=[])


class Tracer:
    """
    Main tracer for emitting structured events and managing spans.

    Thread-safe and safe against exceptions (won't break the run).
    """

    def __init__(self, events_path: str = "traces/events.jsonl", blobs_path: str = "traces/blobs"):
        self.events_path = Path(events_path)
        self.blob_store = BlobStore(blobs_path)
        self._lock = threading.Lock()
        self._last_event_ts = {}  # Track last event timestamp per run_id for dt_ms calculation

        # Ensure directories exist
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def start_run(self, entrypoint: str, args: Optional[Dict] = None, env: Optional[Dict] = None) -> str:
        """
        Start a new run.

        Args:
            entrypoint: Entry point identifier
            args: Run arguments
            env: Environment variables (will be redacted)

        Returns:
            run_id
        """
        run_id = generate_id()
        _run_id.set(run_id)
        _span_stack.set([])
        _span_kind_stack.set([])

        payload = {
            "entrypoint": entrypoint,
            "args": redact_payload(args or {}),
        }
        if env:
            payload["env"] = redact_payload(env)

        self._emit_event(
            run_id=run_id,
            span_id=None,
            parent_span_id=None,
            type="run.start",
            actor="run",
            level="info",
            tags=[],
            payload=payload
        )

        return run_id

    def end_run(self, status: str = "ok", latency_ms: Optional[int] = None):
        """
        End the current run.

        Args:
            status: 'ok' or 'error'
            latency_ms: Total run latency in milliseconds
        """
        run_id = _run_id.get()
        if not run_id:
            return

        payload = {
            "status": status,
        }
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms

        self._emit_event(
            run_id=run_id,
            span_id=None,
            parent_span_id=None,
            type="run.end",
            actor="run",
            level="info" if status == "ok" else "error",
            tags=[],
            payload=payload
        )

    @contextmanager
    def span(self, name: str, actor: str, kind: str = "other", tags: Optional[List[str]] = None):
        """
        Context manager for a span.

        Args:
            name: Span name
            actor: Actor identifier (e.g., 'agent.planner', 'router.core')
            kind: Span kind ('agent', 'router', 'llm', 'tool', 'sandbox', 'function', 'other')
            tags: Optional tags (max 8)

        Yields:
            span_id
        """
        run_id = _run_id.get()
        if not run_id:
            # No active run, skip tracing
            yield None
            return

        span_id = generate_id()
        span_stack = _span_stack.get()
        parent_span_id = span_stack[-1] if span_stack else None

        tags = (tags or [])[:8]  # Max 8 tags

        start_time = time.time()

        # Emit span.start
        self._emit_event(
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            type="span.start",
            actor=actor,
            level="info",
            tags=tags,
            payload={
                "name": name,
                "kind": kind
            }
        )

        # Push span onto stack
        new_stack = span_stack + [span_id]
        _span_stack.set(new_stack)

        # Push kind onto kind stack
        kind_stack = _span_kind_stack.get()
        new_kind_stack = kind_stack + [kind]
        _span_kind_stack.set(new_kind_stack)

        try:
            yield span_id
            status = "ok"
            level = "info"
        except Exception as e:
            status = "error"
            level = "error"
            # Emit exception event
            self.emit_exception(e, actor=actor, tags=tags)
            raise
        finally:
            # Pop span from stack
            _span_stack.set(span_stack)
            # Pop kind from kind stack
            _span_kind_stack.set(kind_stack)

            # Emit span.end
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

            self._emit_event(
                run_id=run_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                type="span.end",
                actor=actor,
                level=level,
                tags=tags,
                payload={
                    "status": status,
                    "latency_ms": latency_ms
                }
            )

    def emit(self, type: str, actor: str, level: str = "info", tags: Optional[List[str]] = None, payload: Optional[Dict] = None):
        """
        Emit a custom event.

        Args:
            type: Event type
            actor: Actor identifier
            level: Log level ('info', 'warn', 'error')
            tags: Optional tags (max 8)
            payload: Event payload
        """
        run_id = _run_id.get()
        if not run_id:
            return

        span_stack = _span_stack.get()
        span_id = span_stack[-1] if span_stack else None
        parent_span_id = span_stack[-2] if len(span_stack) > 1 else None

        tags = (tags or [])[:8]
        payload = redact_payload(payload or {})

        self._emit_event(
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            type=type,
            actor=actor,
            level=level,
            tags=tags,
            payload=payload
        )

    def emit_exception(self, exc: Exception, actor: str = "unknown", tags: Optional[List[str]] = None):
        """
        Emit an exception event.

        Args:
            exc: Exception object
            actor: Actor where exception occurred
            tags: Optional tags
        """
        tb = traceback.extract_tb(exc.__traceback__)
        if tb:
            last_frame = tb[-1]
            where = {
                "file": last_frame.filename,
                "line": last_frame.lineno,
                "func": last_frame.name
            }
        else:
            where = {"file": "unknown", "line": 0}

        stack_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        stack_ref = self.put_blob(stack_trace, "txt")

        payload = {
            "where": where,
            "exc_type": type(exc).__name__,
            "message": str(exc),
            "stack_ref": stack_ref
        }

        # Add span_kind if available (from current span)
        kind_stack = _span_kind_stack.get()
        if kind_stack:
            payload["span_kind"] = kind_stack[-1]

        self.emit(
            type="exception",
            actor=actor,
            level="error",
            tags=tags,
            payload=payload
        )

    def put_blob(self, content: Any, ext: str = "txt") -> str:
        """
        Store content in blob store.

        Args:
            content: Content (string or JSON-serializable object)
            ext: File extension ('txt' or 'json')

        Returns:
            Blob reference (e.g., 'blob:sha256:<hash>')
        """
        try:
            return self.blob_store.put(content, ext)
        except Exception:
            # Safe: return empty ref if blob storage fails
            return "blob:sha256:error"

    def get_blob(self, ref: str) -> str:
        """
        Retrieve blob content by reference.

        Args:
            ref: Blob reference

        Returns:
            Content as string
        """
        return self.blob_store.get(ref)

    def current_run_id(self) -> Optional[str]:
        """Get current run ID."""
        return _run_id.get()

    def current_span_id(self) -> Optional[str]:
        """Get current span ID."""
        span_stack = _span_stack.get()
        return span_stack[-1] if span_stack else None

    def _emit_event(self, run_id: str, span_id: Optional[str], parent_span_id: Optional[str],
                    type: str, actor: str, level: str, tags: List[str], payload: Dict):
        """
        Internal method to emit an event.

        Thread-safe and exception-safe.
        """
        try:
            # Get current timestamp
            ts = time.time()

            # Calculate dt_ms (delta time from last event in this run)
            dt_ms = None
            if run_id in self._last_event_ts:
                dt_ms = int((ts - self._last_event_ts[run_id]) * 1000)

            # Update last event timestamp for this run
            self._last_event_ts[run_id] = ts

            # Generate human-readable timestamps
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)

            # ts_human: for live debugging (HH:MM:SS.sss)
            ts_human = dt.strftime("%H:%M:%S.") + f"{int((ts % 1) * 1000):03d}"

            # ts_iso: for historical logs, incident review (ISO 8601 with timezone)
            ts_iso = dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int((ts % 1) * 1000):03d}Z"

            event = {
                "ts": ts,
                "ts_human": ts_human,
                "ts_iso": ts_iso,
                "run_id": run_id,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "type": type,
                "actor": actor,
                "level": level,
                "tags": tags,
                "payload": payload
            }

            # Add dt_ms only if available (not for first event in run)
            if dt_ms is not None:
                event["dt_ms"] = dt_ms

            # Write to JSONL file (append-only)
            with self._lock:
                with open(self.events_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception:
            # Safe: never break the run
            pass


# Helper functions for LLM calls
def emit_llm_call_start(tracer: Tracer, model: str, prompt: Any, params: Optional[Dict] = None,
                        purpose: Optional[str] = None, actor: str = "llm"):
    """Emit llm.call.start event."""
    # Store prompt in blob
    # If dict/list, save as JSON; otherwise as text
    if isinstance(prompt, (dict, list)):
        prompt_ref = tracer.put_blob(prompt, "json")
        prompt_text = json.dumps(prompt, indent=2)
    else:
        prompt_text = str(prompt)
        prompt_ref = tracer.put_blob(prompt_text, "txt")

    prompt_preview = truncate_preview(prompt_text, 500)

    payload = {
        "model": model,
        "prompt_ref": prompt_ref,
        "prompt_preview": prompt_preview,
    }
    if params:
        payload["params"] = params
    if purpose:
        payload["purpose"] = purpose

    tracer.emit(type="llm.call.start", actor=actor, level="info", payload=payload)


def emit_llm_call_end(tracer: Tracer, response: Any, usage: Optional[Dict] = None,
                      finish_reason: Optional[str] = None, tool_calls: Optional[List[Dict]] = None,
                      actor: str = "llm"):
    """Emit llm.call.end event."""
    # Store response in blob
    # If dict/list, save as JSON; otherwise as text
    if isinstance(response, (dict, list)):
        response_ref = tracer.put_blob(response, "json")
        response_text = json.dumps(response, indent=2)
    else:
        response_text = str(response)
        response_ref = tracer.put_blob(response_text, "txt")

    response_preview = truncate_preview(response_text, 500)

    payload = {
        "response_ref": response_ref,
        "response_preview": response_preview,
    }
    if usage:
        payload["usage"] = usage
    if finish_reason:
        payload["finish_reason"] = finish_reason
    if tool_calls:
        # Store tool call args in blobs
        processed_calls = []
        for tc in tool_calls:
            args_ref = tracer.put_blob(tc.get("args", {}), "json")
            processed_calls.append({
                "name": tc.get("name", "unknown"),
                "args_ref": args_ref
            })
        payload["tool_calls"] = processed_calls

    tracer.emit(type="llm.call.end", actor=actor, level="info", payload=payload)


def emit_tool_call_start(tracer: Tracer, tool: str, args: Any, actor: str = "tool"):
    """Emit tool.call.start event."""
    args_ref = tracer.put_blob(args, "json")

    # Create args preview (truncate if needed)
    if isinstance(args, dict):
        args_preview = {k: (str(v)[:100] + "..." if len(str(v)) > 100 else v)
                       for k, v in list(args.items())[:5]}
    else:
        args_preview = truncate_preview(str(args), 200)

    payload = {
        "tool": tool,
        "args_ref": args_ref,
        "args_preview": args_preview
    }

    tracer.emit(type="tool.call.start", actor=actor, level="info", payload=payload)


def emit_tool_call_end(tracer: Tracer, tool: str, result: Any, status: str = "ok",
                       latency_ms: Optional[int] = None, error: Optional[str] = None,
                       actor: str = "tool"):
    """Emit tool.call.end event."""
    result_ref = tracer.put_blob(result, "json" if isinstance(result, (dict, list)) else "txt")

    payload = {
        "tool": tool,
        "result_ref": result_ref,
        "status": status,
    }
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    if error:
        error_ref = tracer.put_blob(error, "txt")
        payload["error_ref"] = error_ref

    tracer.emit(type="tool.call.end", actor=actor, level="info" if status == "ok" else "error", payload=payload)


def emit_router_decision(tracer: Tracer, rule: str, candidates: List[str], chosen: str,
                        why: Optional[str] = None, actor: str = "router"):
    """Emit router.decision event."""
    payload = {
        "rule": rule,
        "candidates": candidates,
        "chosen": chosen,
    }
    if why:
        why_ref = tracer.put_blob(why, "txt")
        payload["why_ref"] = why_ref

    tracer.emit(type="router.decision", actor=actor, level="info", payload=payload)


def emit_sandbox_exec_start(tracer: Tracer, sandbox: Dict, code: str, inputs: Optional[Any] = None,
                           actor: str = "sandbox"):
    """Emit sandbox.exec.start event."""
    code_ref = tracer.put_blob(code, "txt")

    payload = {
        "sandbox": sandbox,
        "code_ref": code_ref,
    }
    if inputs is not None:
        inputs_ref = tracer.put_blob(inputs, "json")
        payload["inputs_ref"] = inputs_ref

    tracer.emit(type="sandbox.exec.start", actor=actor, level="info", payload=payload)


def emit_sandbox_exec_end(tracer: Tracer, status: str = "ok", stdout: Optional[str] = None,
                         stderr: Optional[str] = None, result: Optional[Any] = None,
                         error: Optional[str] = None, actor: str = "sandbox"):
    """Emit sandbox.exec.end event."""
    payload = {"status": status}

    if stdout:
        payload["stdout_ref"] = tracer.put_blob(stdout, "txt")
    if stderr:
        payload["stderr_ref"] = tracer.put_blob(stderr, "txt")
    if result is not None:
        payload["result_ref"] = tracer.put_blob(result, "json" if isinstance(result, (dict, list)) else "txt")
    if error:
        payload["error_ref"] = tracer.put_blob(error, "txt")

    tracer.emit(type="sandbox.exec.end", actor=actor, level="info" if status == "ok" else "error", payload=payload)
