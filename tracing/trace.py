"""Convenience module for tracing - re-exports main API."""

from .tracer import (
    Tracer,
    emit_llm_call_end,
    emit_llm_call_start,
    emit_router_decision,
    emit_sandbox_exec_end,
    emit_sandbox_exec_start,
    emit_tool_call_end,
    emit_tool_call_start,
)
from . import get_tracer, set_tracer

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
]
