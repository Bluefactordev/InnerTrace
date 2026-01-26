"""LLM Execution Observability - Structured tracing for LLM orchestration."""

from .tracer import (
    STORYTELLING_EVENT_TYPES,
    STORYTELLING_EVENT_TYPES_STABLE,
    STORYTELLING_EVENT_TYPES_EXPERIMENTAL,
    Tracer,
    emit_code_execution_error,
    emit_llm_call_end,
    emit_llm_call_start,
    emit_router_decision,
    emit_sandbox_exec_end,
    emit_sandbox_exec_start,
    emit_tool_call_end,
    emit_tool_call_start,
)

# Global tracer instance
_global_tracer = None


def get_tracer() -> Tracer:
    """Get or create global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def set_tracer(tracer: Tracer):
    """Set global tracer instance."""
    global _global_tracer
    _global_tracer = tracer


__all__ = [
    "Tracer",
    "get_tracer",
    "set_tracer",
    "emit_llm_call_start",
    "emit_llm_call_end",
    "emit_tool_call_start",
    "emit_tool_call_end",
    "emit_router_decision",
    "emit_sandbox_exec_start",
    "emit_sandbox_exec_end",
    "emit_code_execution_error",
    "STORYTELLING_EVENT_TYPES",
    "STORYTELLING_EVENT_TYPES_STABLE",
    "STORYTELLING_EVENT_TYPES_EXPERIMENTAL",
]
