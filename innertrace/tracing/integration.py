"""
Integration hooks for adding tracing to existing codebase.

This module provides wrapper functions and decorators that can be used
to add tracing to key integration points without requiring extensive refactoring.

IMPORTANT: All wrappers create dedicated spans for each call (llm/tool/sandbox).
"""

import functools
import time
from typing import Any, Callable, Dict, List, Optional

from . import get_tracer
from .tracer import (
    emit_llm_call_end,
    emit_llm_call_start,
    emit_router_decision,
    emit_sandbox_exec_end,
    emit_sandbox_exec_start,
    emit_tool_call_end,
    emit_tool_call_start,
)


async def traced_llm_generate(
    call_fn: Callable,
    model_id: str,
    messages: Any,
    params: Optional[Dict] = None,
    purpose: Optional[str] = None,
    actor: str = "llm"
) -> Any:
    """
    Wrapper for tracing async LLM generate calls.

    Creates a dedicated span for the LLM call and emits llm.call.start/end events.
    Provider-agnostic: accepts a coroutine factory (lambda) that executes the actual call.

    Args:
        call_fn: Coroutine factory (e.g., lambda: llm.ainvoke(messages))
        model_id: Model identifier
        messages: Messages to send (for logging only)
        params: Call parameters (for logging only)
        purpose: Purpose of the call (for logging only)
        actor: Actor identifier

    Usage:
        response = await traced_llm_generate(
            lambda: llm.ainvoke(messages),
            model_id="gpt-4",
            messages=messages,
            params={"temperature": 0.7},
            purpose="reasoning"
        )
    """
    tracer = get_tracer()

    # Skip if no active run
    if not tracer.current_run_id():
        return await call_fn()

    # Create dedicated span for this LLM call
    with tracer.span(f"llm.call", actor=actor, kind="llm", tags=[model_id, purpose] if purpose else [model_id]):
        # Emit start event
        emit_llm_call_start(tracer, model_id, messages, params, purpose, actor=actor)

        try:
            # Call the LLM (execute the factory)
            response = await call_fn()

            # Extract usage if available
            usage = None
            finish_reason = None
            tool_calls = None

            if hasattr(response, 'usage_metadata'):
                usage = {
                    "input_tokens": getattr(response.usage_metadata, 'input_tokens', 0),
                    "output_tokens": getattr(response.usage_metadata, 'output_tokens', 0),
                }

            if hasattr(response, 'response_metadata'):
                finish_reason = response.response_metadata.get('finish_reason')

            # Emit end event (success)
            emit_llm_call_end(tracer, response, usage, finish_reason, tool_calls, actor=actor)

            return response

        except Exception as e:
            # Emit end event (error) - spec doesn't have status field, so we emit exception only
            # The span.end will have status=error automatically
            tracer.emit_exception(e, actor=actor)
            raise


def traced_tool_call(tool_name: str, actor: str = "tool"):
    """
    Decorator for tracing tool calls.

    Creates a dedicated span for the tool call and emits tool.call.start/end events.

    Usage:
        @traced_tool_call("search_web")
        async def search_web(query: str):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()

            # Skip if no active run
            if not tracer.current_run_id():
                return await func(*args, **kwargs)

            # Create dedicated span for this tool call
            with tracer.span(f"tool.call:{tool_name}", actor=actor, kind="tool", tags=[tool_name]):
                # Prepare args dict
                args_dict = {}
                if args:
                    args_dict["args"] = args
                if kwargs:
                    args_dict.update(kwargs)

                # Emit start event
                emit_tool_call_start(tracer, tool_name, args_dict, actor=actor)

                start_time = time.time()
                status = "ok"
                result = None
                error = None

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    error = str(e)
                    tracer.emit_exception(e, actor=actor)
                    raise
                finally:
                    latency_ms = int((time.time() - start_time) * 1000)
                    emit_tool_call_end(tracer, tool_name, result, status, latency_ms, error, actor=actor)

        return wrapper
    return decorator


def trace_router_decision(rule: str, candidates: List[str], chosen: str, why: Optional[str] = None, actor: str = "router"):
    """
    Helper to emit router decision event.

    Usage:
        trace_router_decision(
            rule="complexity_based",
            candidates=["simple", "complex"],
            chosen="complex",
            why="Query requires multi-step reasoning"
        )
    """
    tracer = get_tracer()
    if tracer.current_run_id():
        emit_router_decision(tracer, rule, candidates, chosen, why, actor=actor)


async def traced_sandbox_exec(
    exec_callable: Callable,
    code: str,
    sandbox_info: Optional[Dict] = None,
    inputs: Optional[Any] = None,
    actor: str = "sandbox",
    **kwargs
) -> Any:
    """
    Wrapper for tracing sandbox execution.

    Creates a dedicated span for the sandbox execution and emits sandbox.exec.start/end events.

    Usage:
        result = await traced_sandbox_exec(
            sandbox.execute,
            code="print('hello')",
            sandbox_info={"kind": "python", "image": "python:3.11"},
            inputs={"x": 42}
        )
    """
    tracer = get_tracer()

    # Skip if no active run
    if not tracer.current_run_id():
        return await exec_callable(code, **kwargs)

    sandbox = sandbox_info or {"kind": "unknown"}

    # Create dedicated span for this sandbox execution
    with tracer.span(f"sandbox.exec", actor=actor, kind="sandbox", tags=[sandbox.get("kind", "unknown")]):
        # Emit start event
        emit_sandbox_exec_start(tracer, sandbox, code, inputs, actor=actor)

        status = "ok"
        result = None
        stdout = None
        stderr = None
        error = None

        try:
            result = await exec_callable(code, **kwargs)

            # Try to extract stdout/stderr if available
            if isinstance(result, dict):
                stdout = result.get("stdout")
                stderr = result.get("stderr")

            return result

        except Exception as e:
            status = "error"
            error = str(e)
            tracer.emit_exception(e, actor=actor)
            raise

        finally:
            emit_sandbox_exec_end(tracer, status, stdout, stderr, result, error, actor=actor)


# Convenience function to start a traced run
def start_traced_run(
    entrypoint: str,
    args: Optional[Dict] = None,
    env: Optional[Dict] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Start a traced run.

    Usage:
        run_id = start_traced_run("api.chat", {"query": "hello"})
        try:
            # ... do work ...
            end_traced_run("ok")
        except Exception:
            end_traced_run("error")
    """
    tracer = get_tracer()
    return tracer.start_run(entrypoint, args, env, seed=seed)


def end_traced_run(status: str = "ok", latency_ms: Optional[int] = None):
    """End the current traced run."""
    tracer = get_tracer()
    tracer.end_run(status, latency_ms)
