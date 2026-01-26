"""Declarative function-level tracing for innertrace.

This module provides decorators and utilities for tracing function calls
without global hooks or monkey patching.
"""

import functools
import inspect
import time
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Callable, Dict, List, Optional
from fnmatch import fnmatch

from .serialize import serialize_safe
from .tracer import _run_id, _span_stack, _span_kind_stack, generate_id


def trace_function(
    name: Optional[str] = None,
    capture_args: bool = True,
    capture_return: bool = False,
    redact: Optional[List[str]] = None,
    max_repr: int = 500,
    max_items: int = 50,
) -> Callable:
    """
    Decorator to trace function execution.

    Emits span.start and span.end events with function metadata, arguments,
    and optionally return values. Works with both sync and async functions.

    Args:
        name: Custom name for the span (default: module.qualname)
        capture_args: Whether to capture function arguments (default: True)
        capture_return: Whether to capture return value (default: False)
        redact: Additional keys to redact beyond standard secrets
        max_repr: Maximum length for repr() of objects (default: 500)
        max_items: Maximum items in lists/dicts (default: 50)

    Returns:
        Decorated function

    Example:
        @trace_function(capture_args=True, capture_return=False)
        def my_function(x, y):
            return x + y

        @trace_function(name="custom_async_fn")
        async def async_function(data):
            return await process(data)
    """
    def decorator(func: Callable) -> Callable:
        # Determine effective name
        effective_name = name or f"{func.__module__}.{func.__qualname__}"
        func_qualname = f"{func.__module__}.{func.__qualname__}"

        # Handle async functions
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get tracer - avoid circular import
                from . import get_tracer
                tracer = get_tracer()

                # Check if we're in a run
                run_id = _run_id.get()
                if not run_id:
                    # No active run, execute without tracing
                    return await func(*args, **kwargs)

                # Generate span ID and get parent
                span_id = generate_id()
                span_stack = _span_stack.get()
                parent_span_id = span_stack[-1] if span_stack else None

                # Build payload for span.start
                payload = {
                    "name": effective_name,
                    "kind": "function",
                    "func": func_qualname,
                }
                if capture_args:
                    payload["args"] = serialize_safe(args, max_repr, max_items, redact)
                    payload["kwargs"] = serialize_safe(kwargs, max_repr, max_items, redact)

                # Emit span.start
                start_time = time.time()
                tracer._emit_event(
                    run_id=run_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    type="span.start",
                    actor=f"function.{func.__name__}",
                    level="info",
                    tags=[],
                    payload=payload
                )

                # Push span onto stack
                _span_stack.set(span_stack + [span_id])
                _span_kind_stack.set(_span_kind_stack.get() + ["function"])

                # Execute function
                status = "ok"
                level = "info"
                result = None
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    level = "error"
                    # Emit exception event
                    tracer.emit_exception(e, actor=f"function.{func.__name__}")
                    raise
                finally:
                    # Pop span from stack
                    _span_stack.set(span_stack)
                    _span_kind_stack.set(_span_kind_stack.get()[:-1])

                    # Emit span.end
                    end_payload = {
                        "status": status,
                        "latency_ms": int((time.time() - start_time) * 1000)
                    }
                    if capture_return and status == "ok" and result is not None:
                        end_payload["return"] = serialize_safe(result, max_repr, max_items, redact)

                    tracer._emit_event(
                        run_id=run_id,
                        span_id=span_id,
                        parent_span_id=parent_span_id,
                        type="span.end",
                        actor=f"function.{func.__name__}",
                        level=level,
                        tags=[],
                        payload=end_payload
                    )

            async_wrapper.__innertrace_wrapped__ = True
            return async_wrapper

        # Handle sync functions
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Get tracer - avoid circular import
                from . import get_tracer
                tracer = get_tracer()

                # Check if we're in a run
                run_id = _run_id.get()
                if not run_id:
                    # No active run, execute without tracing
                    return func(*args, **kwargs)

                # Generate span ID and get parent
                span_id = generate_id()
                span_stack = _span_stack.get()
                parent_span_id = span_stack[-1] if span_stack else None

                # Build payload for span.start
                payload = {
                    "name": effective_name,
                    "kind": "function",
                    "func": func_qualname,
                }
                if capture_args:
                    payload["args"] = serialize_safe(args, max_repr, max_items, redact)
                    payload["kwargs"] = serialize_safe(kwargs, max_repr, max_items, redact)

                # Emit span.start
                start_time = time.time()
                tracer._emit_event(
                    run_id=run_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    type="span.start",
                    actor=f"function.{func.__name__}",
                    level="info",
                    tags=[],
                    payload=payload
                )

                # Push span onto stack
                _span_stack.set(span_stack + [span_id])
                _span_kind_stack.set(_span_kind_stack.get() + ["function"])

                # Execute function
                status = "ok"
                level = "info"
                result = None
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    level = "error"
                    # Emit exception event
                    tracer.emit_exception(e, actor=f"function.{func.__name__}")
                    raise
                finally:
                    # Pop span from stack
                    _span_stack.set(span_stack)
                    _span_kind_stack.set(_span_kind_stack.get()[:-1])

                    # Emit span.end
                    end_payload = {
                        "status": status,
                        "latency_ms": int((time.time() - start_time) * 1000)
                    }
                    if capture_return and status == "ok" and result is not None:
                        end_payload["return"] = serialize_safe(result, max_repr, max_items, redact)

                    tracer._emit_event(
                        run_id=run_id,
                        span_id=span_id,
                        parent_span_id=parent_span_id,
                        type="span.end",
                        actor=f"function.{func.__name__}",
                        level=level,
                        tags=[],
                        payload=end_payload
                    )

            sync_wrapper.__innertrace_wrapped__ = True
            return sync_wrapper

    return decorator


@contextmanager
def trace_block(name: str, payload: Optional[Dict[str, Any]] = None):
    """
    Context manager for tracing a code block.

    Emits span.start and span.end events with custom name and payload.
    Works only in synchronous contexts.

    Args:
        name: Name for the block span
        payload: Optional custom payload dict

    Example:
        with trace_block("data_processing", payload={"records": 100}):
            process_data()
    """
    # Get tracer - avoid circular import
    from . import get_tracer
    tracer = get_tracer()

    # Check if we're in a run
    run_id = _run_id.get()
    if not run_id:
        # No active run, execute without tracing
        yield
        return

    # Generate span ID and get parent
    span_id = generate_id()
    span_stack = _span_stack.get()
    parent_span_id = span_stack[-1] if span_stack else None

    # Build payload for span.start
    span_payload = {
        "name": name,
        "kind": "function",
    }
    if payload:
        span_payload.update(payload)

    # Emit span.start
    start_time = time.time()
    tracer._emit_event(
        run_id=run_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        type="span.start",
        actor=f"block.{name}",
        level="info",
        tags=[],
        payload=span_payload
    )

    # Push span onto stack
    _span_stack.set(span_stack + [span_id])
    _span_kind_stack.set(_span_kind_stack.get() + ["function"])

    # Execute block
    status = "ok"
    level = "info"
    try:
        yield
    except Exception as e:
        status = "error"
        level = "error"
        # Emit exception event
        tracer.emit_exception(e, actor=f"block.{name}")
        raise
    finally:
        # Pop span from stack
        _span_stack.set(span_stack)
        _span_kind_stack.set(_span_kind_stack.get()[:-1])

        # Emit span.end
        end_payload = {
            "status": status,
            "latency_ms": int((time.time() - start_time) * 1000)
        }

        tracer._emit_event(
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            type="span.end",
            actor=f"block.{name}",
            level=level,
            tags=[],
            payload=end_payload
        )


@asynccontextmanager
async def trace_block_async(name: str, payload: Optional[Dict[str, Any]] = None):
    """
    Async context manager for tracing a code block.

    Emits span.start and span.end events with custom name and payload.
    Works only in asynchronous contexts.

    Args:
        name: Name for the block span
        payload: Optional custom payload dict

    Example:
        async with trace_block_async("async_processing", payload={"items": 50}):
            await process_async()
    """
    # Get tracer - avoid circular import
    from . import get_tracer
    tracer = get_tracer()

    # Check if we're in a run
    run_id = _run_id.get()
    if not run_id:
        # No active run, execute without tracing
        yield
        return

    # Generate span ID and get parent
    span_id = generate_id()
    span_stack = _span_stack.get()
    parent_span_id = span_stack[-1] if span_stack else None

    # Build payload for span.start
    span_payload = {
        "name": name,
        "kind": "function",
    }
    if payload:
        span_payload.update(payload)

    # Emit span.start
    start_time = time.time()
    tracer._emit_event(
        run_id=run_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        type="span.start",
        actor=f"block.{name}",
        level="info",
        tags=[],
        payload=span_payload
    )

    # Push span onto stack
    _span_stack.set(span_stack + [span_id])
    _span_kind_stack.set(_span_kind_stack.get() + ["function"])

    # Execute block
    status = "ok"
    level = "info"
    try:
        yield
    except Exception as e:
        status = "error"
        level = "error"
        # Emit exception event
        tracer.emit_exception(e, actor=f"block.{name}")
        raise
    finally:
        # Pop span from stack
        _span_stack.set(span_stack)
        _span_kind_stack.set(_span_kind_stack.get()[:-1])

        # Emit span.end
        end_payload = {
            "status": status,
            "latency_ms": int((time.time() - start_time) * 1000)
        }

        tracer._emit_event(
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            type="span.end",
            actor=f"block.{name}",
            level=level,
            tags=[],
            payload=end_payload
        )


# 🔧 GLOBAL FILTER CONFIGURATION
# Permette di definire filtri globali che vengono applicati a tutte le chiamate trace_module()
# Può essere sovrascritto da variabili d'ambiente o file di configurazione
_GLOBAL_TRACE_EXCLUDE = []
_GLOBAL_TRACE_INCLUDE = []


def set_global_trace_filters(
    exclude: Optional[List[str]] = None,
    include: Optional[List[str]] = None
) -> None:
    """
    Imposta i filtri globali di tracing.
    
    Args:
        exclude: Lista di pattern (fnmatch) da escludere globalmente
        include: Lista di pattern (fnmatch) da includere globalmente
    """
    global _GLOBAL_TRACE_EXCLUDE, _GLOBAL_TRACE_INCLUDE
    if exclude is not None:
        _GLOBAL_TRACE_EXCLUDE = exclude.copy()
    if include is not None:
        _GLOBAL_TRACE_INCLUDE = include.copy()


def get_global_trace_filters() -> dict:
    """
    Restituisce i filtri globali di tracing.
    
    Returns:
        dict con chiavi 'exclude' e 'include'
    """
    return {
        'exclude': _GLOBAL_TRACE_EXCLUDE.copy(),
        'include': _GLOBAL_TRACE_INCLUDE.copy()
    }


def trace_module(
    globals_dict: Dict[str, Any],
    *,
    include_private: bool = False,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    redact: Optional[List[str]] = None,
    name_prefix: Optional[str] = None,
) -> None:
    """
    Instrument all functions in a module with tracing.

    Call this at the end of your module file to automatically wrap all
    functions defined in the module.

    Args:
        globals_dict: The module's globals() dict
        include_private: Whether to include private functions (starting with _)
        include: List of function name patterns to include (fnmatch patterns)
        exclude: List of function name patterns to exclude (fnmatch patterns)
        redact: Additional keys to redact in function arguments
        name_prefix: Prefix for span names (default: module name)

    Example:
        # At end of mymodule.py:
        from innertrace import trace_module

        def public_function():
            pass

        def _private_function():
            pass

        trace_module(globals(), include_private=False)
    """
    module_name = globals_dict.get("__name__", "<unknown>")

    # 🔧 MERGE GLOBAL FILTERS: Unisci filtri globali con quelli locali
    global_filters = get_global_trace_filters()
    effective_exclude = global_filters['exclude'].copy()
    if exclude:
        effective_exclude.extend(exclude)
    
    effective_include = global_filters['include'].copy()
    if include:
        effective_include.extend(include)
    effective_include = effective_include if effective_include else None

    # Iterate through all items in globals
    for name, obj in list(globals_dict.items()):
        # Check if it's a function
        if not inspect.isfunction(obj):
            continue

        # Check if it's defined in this module
        if obj.__module__ != module_name:
            continue

        # Skip if already wrapped
        if getattr(obj, "__innertrace_wrapped__", False):
            continue

        # Skip private functions unless explicitly included
        if name.startswith("_") and not include_private:
            continue

        # Apply include patterns (global + local)
        if effective_include:
            if not any(fnmatch(name, pattern) for pattern in effective_include):
                continue

        # Apply exclude patterns (global + local)
        if effective_exclude:
            if any(fnmatch(name, pattern) for pattern in effective_exclude):
                continue

        # Determine effective name
        if name_prefix:
            effective_name = f"{name_prefix}.{name}"
        else:
            effective_name = None  # Let trace_function use default

        # Wrap the function
        wrapped = trace_function(
            name=effective_name,
            capture_args=True,
            capture_return=False,
            redact=redact
        )(obj)

        # Update globals dict
        globals_dict[name] = wrapped
