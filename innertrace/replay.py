"""Replay manager for deterministic trace replays."""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    from freezegun import freeze_time
except ImportError:  # pragma: no cover - optional dependency
    freeze_time = None

from .tracing import get_tracer
from .tracing.blob_store import BlobStore
from .tracing.tracer import emit_tool_call_end, emit_tool_call_start


class NonDeterminismDetected(RuntimeError):
    """Raised when replay cannot match recorded events."""


class ReplayToolError(RuntimeError):
    """Raised when a recorded tool call failed during replay."""


class EventStore:
    """Base interface for loading events."""

    def load_events(self, run_id: str) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError


class JsonlEventStore(EventStore):
    """JSONL event store (default)."""

    def __init__(self, events_dir: str = "traces/events"):
        self.events_dir = Path(events_dir)

    def load_events(self, run_id: str) -> Iterable[Dict[str, Any]]:
        events_path = self.events_dir / f"{run_id}.jsonl"
        if not events_path.exists():
            raise FileNotFoundError(f"Replay log not found: {events_path}")
        with open(events_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


@dataclass
class _ReplayCall:
    start: Optional[Dict[str, Any]]
    end: Dict[str, Any]


class ReplayManager:
    """Loads and serves replay events for a specific run."""

    def __init__(
        self,
        run_id: str,
        *,
        events_dir: str = "traces/events",
        blobs_dir: str = "traces/blobs",
        event_store: Optional[EventStore] = None,
        validate_prompts: bool = False,
        validate_tool_args: bool = False,
    ):
        self.run_id = run_id
        self.event_store = event_store or JsonlEventStore(events_dir)
        self.blob_store = BlobStore(blobs_dir)
        self.validate_prompts = validate_prompts
        self.validate_tool_args = validate_tool_args
        self._llm_calls: List[_ReplayCall] = []
        self._tool_calls: List[_ReplayCall] = []
        self._llm_cursor = 0
        self._tool_cursor = 0
        self._run_start_iso: Optional[str] = None
        self._random_seed: Optional[int] = None
        self._load_events()

    @property
    def run_start_iso(self) -> Optional[str]:
        return self._run_start_iso

    @property
    def random_seed(self) -> Optional[int]:
        return self._random_seed

    def _load_events(self) -> None:
        pending_llm: deque[Dict[str, Any]] = deque()
        pending_tool: deque[Dict[str, Any]] = deque()

        for event in self.event_store.load_events(self.run_id):
            event_type = event.get("type")
            if event_type == "run.start" and self._run_start_iso is None:
                self._run_start_iso = event.get("ts_iso")
                payload = event.get("payload", {})
                self._random_seed = payload.get("random_seed") or payload.get("seed")
            elif event_type == "run.seed":
                payload = event.get("payload", {})
                seed = payload.get("seed")
                if seed is not None:
                    self._random_seed = seed
            elif event_type == "llm.call.start":
                pending_llm.append(event)
            elif event_type == "llm.call.end":
                start_event = pending_llm.popleft() if pending_llm else None
                self._llm_calls.append(_ReplayCall(start=start_event, end=event))
            elif event_type == "tool.call.start":
                pending_tool.append(event)
            elif event_type == "tool.call.end":
                start_event = pending_tool.popleft() if pending_tool else None
                self._tool_calls.append(_ReplayCall(start=start_event, end=event))

    def next_llm_response(self, prompt: Any, *, model: Optional[str] = None) -> Any:
        if self._llm_cursor >= len(self._llm_calls):
            raise NonDeterminismDetected("No more recorded LLM calls available.")
        record = self._llm_calls[self._llm_cursor]
        self._llm_cursor += 1

        start_payload = (record.start or {}).get("payload", {})
        if model and start_payload.get("model") and model != start_payload.get("model"):
            raise NonDeterminismDetected(
                f"LLM model mismatch: expected {start_payload.get('model')}, got {model}"
            )

        if self.validate_prompts and record.start:
            recorded_prompt = self._load_prompt(start_payload)
            if recorded_prompt is not None:
                current_prompt = self._normalize_prompt(prompt)
                if current_prompt != recorded_prompt:
                    raise NonDeterminismDetected("Prompt mismatch detected during replay.")

        end_payload = record.end.get("payload", {})
        response_ref = end_payload.get("response_ref")
        if response_ref:
            return self._load_blob(response_ref)
        return end_payload.get("response_preview")

    def next_tool_result(self, tool: str, args: Optional[Dict[str, Any]] = None) -> Any:
        if self._tool_cursor >= len(self._tool_calls):
            raise NonDeterminismDetected("No more recorded tool calls available.")
        record = self._tool_calls[self._tool_cursor]
        self._tool_cursor += 1

        start_payload = (record.start or {}).get("payload", {})
        if start_payload.get("tool") and start_payload.get("tool") != tool:
            raise NonDeterminismDetected(
                f"Tool mismatch: expected {start_payload.get('tool')}, got {tool}"
            )
        if self.validate_tool_args and args is not None and record.start:
            recorded_args = self._load_tool_args(start_payload)
            if recorded_args is not None:
                if self._normalize_value(args) != self._normalize_value(recorded_args):
                    raise NonDeterminismDetected("Tool args mismatch detected during replay.")

        end_payload = record.end.get("payload", {})
        status = end_payload.get("status")
        if status and status != "ok":
            error_ref = end_payload.get("error_ref")
            message = self._load_blob(error_ref) if error_ref else "Recorded tool error"
            raise ReplayToolError(f"Tool failed during recorded run: {message}")

        result_ref = end_payload.get("result_ref")
        if result_ref:
            return self._load_blob(result_ref)
        return None

    def _load_prompt(self, payload: Dict[str, Any]) -> Optional[str]:
        prompt_ref = payload.get("prompt_ref")
        if prompt_ref:
            return self._normalize_prompt(self._load_blob(prompt_ref))
        prompt_preview = payload.get("prompt_preview")
        if prompt_preview:
            return self._normalize_prompt(prompt_preview)
        return None

    def _load_tool_args(self, payload: Dict[str, Any]) -> Optional[Any]:
        args_ref = payload.get("args_ref")
        if args_ref:
            return self._load_blob(args_ref)
        return payload.get("args_preview")

    def _load_blob(self, ref: str) -> Any:
        if not ref:
            return None
        content = self.blob_store.get(ref)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content

    @staticmethod
    def _normalize_value(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        return str(value)

    def _normalize_prompt(self, prompt: Any) -> str:
        return self._normalize_value(prompt)


_ACTIVE_REPLAY_MANAGER: contextvars.ContextVar[Optional[ReplayManager]] = contextvars.ContextVar(
    "replay_manager",
    default=None,
)


def get_replay_manager() -> Optional[ReplayManager]:
    return _ACTIVE_REPLAY_MANAGER.get()


def _set_replay_manager(manager: Optional[ReplayManager]) -> None:
    _ACTIVE_REPLAY_MANAGER.set(manager)


class _OpenAIPatch:
    def __init__(self, patches: List[Tuple[Any, str, Any]]):
        self._patches = patches

    def restore(self) -> None:
        for obj, attr, original in self._patches:
            setattr(obj, attr, original)


class _ReplayOpenAIChoice:
    def __init__(self, content: str):
        self.message = type("ReplayMessage", (), {"content": content})


class _ReplayOpenAIResponse:
    def __init__(self, content: str):
        self.choices = [_ReplayOpenAIChoice(content)]


def _wrap_openai_response(result: Any) -> Any:
    if isinstance(result, dict) and "choices" in result:
        return result
    if hasattr(result, "choices"):
        return result
    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return _ReplayOpenAIResponse(content)


def _patch_openai(manager: ReplayManager) -> Optional[_OpenAIPatch]:
    try:
        import openai
    except ImportError:  # pragma: no cover - optional dependency
        return None

    patches: List[Tuple[Any, str, Any]] = []

    def wrap_sync():
        def wrapper(self, *args, **kwargs):
            prompt = kwargs.get("messages") or kwargs.get("prompt")
            model = kwargs.get("model")
            result = manager.next_llm_response(prompt, model=model)
            return _wrap_openai_response(result)

        return wrapper

    def wrap_async():
        async def wrapper(self, *args, **kwargs):
            prompt = kwargs.get("messages") or kwargs.get("prompt")
            model = kwargs.get("model")
            result = manager.next_llm_response(prompt, model=model)
            return _wrap_openai_response(result)

        return wrapper

    def patch_target(obj, attr, wrapper_factory):
        if obj and hasattr(obj, attr):
            original = getattr(obj, attr)
            setattr(obj, attr, wrapper_factory())
            patches.append((obj, attr, original))

    resources = getattr(openai, "resources", None)
    if resources:
        chat = getattr(resources, "chat", None)
        if chat:
            completions = getattr(chat, "completions", None)
            if completions:
                patch_target(getattr(completions, "Completions", None), "create", wrap_sync)
                patch_target(getattr(completions, "AsyncCompletions", None), "create", wrap_async)

    patch_target(getattr(openai, "ChatCompletion", None), "create", wrap_sync)

    if patches:
        return _OpenAIPatch(patches)
    return None


def in_replay_mode() -> bool:
    return _ACTIVE_REPLAY_MANAGER.get() is not None


def replay(run_id: str, **kwargs) -> "ReplayContext":
    return ReplayContext(run_id=run_id, **kwargs)


class ReplayContext:
    """Context manager enabling deterministic replay for a run."""

    def __init__(
        self,
        run_id: str,
        *,
        events_dir: str = "traces/events",
        blobs_dir: str = "traces/blobs",
        validate_prompts: bool = False,
        validate_tool_args: bool = False,
    ):
        self.manager = ReplayManager(
            run_id,
            events_dir=events_dir,
            blobs_dir=blobs_dir,
            validate_prompts=validate_prompts,
            validate_tool_args=validate_tool_args,
        )
        self._freezer = None
        self._openai_patch: Optional[_OpenAIPatch] = None
        self._random_state = None
        self._numpy_state = None
        self._manager_token = None

    def __enter__(self) -> ReplayManager:
        self._manager_token = _ACTIVE_REPLAY_MANAGER.set(self.manager)
        self._openai_patch = _patch_openai(self.manager)

        if self.manager.run_start_iso and freeze_time:
            self._freezer = freeze_time(self.manager.run_start_iso)
            self._freezer.start()

        self._random_state = random.getstate()
        if np:
            self._numpy_state = np.random.get_state()

        seed = self.manager.random_seed
        if seed is not None:
            random.seed(seed)
            if np:
                np.random.seed(seed)

        return self.manager

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._freezer:
            self._freezer.stop()
        if self._openai_patch:
            self._openai_patch.restore()
        if self._random_state is not None:
            random.setstate(self._random_state)
        if np and self._numpy_state is not None:
            np.random.set_state(self._numpy_state)
        if self._manager_token is not None:
            _ACTIVE_REPLAY_MANAGER.reset(self._manager_token)


def tool(name: Optional[str] = None, actor: str = "tool"):
    """Decorator for tool calls with replay support.

    In replay mode, the tool execution is skipped and recorded output is returned.
    """

    def decorator(func):
        tool_name = name or func.__name__
        if hasattr(func, "__innertrace_tool_wrapped__"):
            return func

        def build_args_dict(args, kwargs) -> Dict[str, Any]:
            args_dict: Dict[str, Any] = {}
            if args:
                args_dict["args"] = args
            if kwargs:
                args_dict.update(kwargs)
            return args_dict

        @functools.wraps(func)
        async def _async_execute(*args, **kwargs):
            tracer = get_tracer()
            if in_replay_mode():
                manager = get_replay_manager()
                if not manager:
                    raise NonDeterminismDetected("Replay manager unavailable.")
                return manager.next_tool_result(tool_name, build_args_dict(args, kwargs))

            if not tracer.current_run_id():
                return await func(*args, **kwargs)

            with tracer.span(f"tool.call:{tool_name}", actor=actor, kind="tool", tags=[tool_name]):
                args_dict = build_args_dict(args, kwargs)
                emit_tool_call_start(tracer, tool_name, args_dict, actor=actor)
                start_time = time.time()
                status = "ok"
                result = None
                error = None
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as exc:
                    status = "error"
                    error = str(exc)
                    tracer.emit_exception(exc, actor=actor)
                    raise
                finally:
                    latency_ms = int((time.time() - start_time) * 1000)
                    emit_tool_call_end(tracer, tool_name, result, status, latency_ms, error, actor=actor)

        @functools.wraps(func)
        def _sync_execute(*args, **kwargs):
            tracer = get_tracer()
            if in_replay_mode():
                manager = get_replay_manager()
                if not manager:
                    raise NonDeterminismDetected("Replay manager unavailable.")
                return manager.next_tool_result(tool_name, build_args_dict(args, kwargs))

            if not tracer.current_run_id():
                return func(*args, **kwargs)

            with tracer.span(f"tool.call:{tool_name}", actor=actor, kind="tool", tags=[tool_name]):
                args_dict = build_args_dict(args, kwargs)
                emit_tool_call_start(tracer, tool_name, args_dict, actor=actor)
                start_time = time.time()
                status = "ok"
                result = None
                error = None
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as exc:
                    status = "error"
                    error = str(exc)
                    tracer.emit_exception(exc, actor=actor)
                    raise
                finally:
                    latency_ms = int((time.time() - start_time) * 1000)
                    emit_tool_call_end(tracer, tool_name, result, status, latency_ms, error, actor=actor)

        wrapper = _async_execute if inspect.iscoroutinefunction(func) else _sync_execute
        wrapper.__innertrace_tool_wrapped__ = True
        return wrapper

    return decorator


def replay_tool(func: Optional[Any] = None, *, name: Optional[str] = None, actor: str = "tool"):
    """Decorator alias for replay-aware tool calls."""
    if func is None:
        return tool(name=name, actor=actor)
    return tool(name=name or func.__name__, actor=actor)(func)


__all__ = [
    "ReplayContext",
    "ReplayManager",
    "NonDeterminismDetected",
    "ReplayToolError",
    "replay",
    "tool",
    "replay_tool",
    "record_seed",
    "in_replay_mode",
]


def record_seed(seed: int) -> None:
    """Record the initial random seed in the current run."""
    tracer = get_tracer()
    if tracer.current_run_id():
        tracer.emit(type="run.seed", actor="run", level="info", payload={"seed": seed})
