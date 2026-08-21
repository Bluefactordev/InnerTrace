# Usage and integration guide

This guide covers the supported package paths beyond the README Quick Start.

## Install

Install the released package or a locally built wheel:

```bash
python -m pip install innertrace==0.3.1
# or
python -m pip install dist/innertrace-0.3.1-py3-none-any.whl
```

For development, use `python -m pip install -e '.[dev]'`.

## Run lifecycle and spans

Create one `Tracer` per storage destination, start a run, and close it with its
final status:

```python
from innertrace import Tracer

tracer = Tracer("traces/events.jsonl", "traces/blobs")
run_id = tracer.start_run("assistant.answer", {"model": "local-example"})

try:
    with tracer.span("plan", actor="agent.planner", kind="agent"):
        pass
except Exception:
    tracer.end_run("error")
    raise
else:
    tracer.end_run("ok")
```

Nested spans establish `parent_span_id` links. A span records an `exception`
and ends with `status=error` when its body raises; the original exception is
re-raised.

## Event helpers

The public helpers in `innertrace.tracing` cover model calls, tools, routing,
and sandbox execution:

```python
from innertrace.tracing import (
    emit_llm_call_end,
    emit_llm_call_start,
    emit_router_decision,
    emit_sandbox_exec_end,
    emit_sandbox_exec_start,
    emit_tool_call_end,
    emit_tool_call_start,
)
```

Use `Tracer.emit(...)` for application events with a clear namespace such as
`biz.*`. Keep payloads compact and avoid raw sensitive content.

`innertrace.tracing.integration` also provides async wrappers for existing LLM,
tool, router, and sandbox boundaries. They use the same tracer and event
vocabulary; no framework integration is required.

## Read-only projections

The JSONL stream remains the source of truth. These helpers derive views without
rewriting it:

```python
from innertrace.tracing.projections import (
    compact_run_view,
    failure_context_view,
    find_last_run,
    list_runs,
    llm_call_view,
    timeline_view,
    tool_chain_view,
)
```

- `timeline_view` preserves the complete span hierarchy.
- `compact_run_view` selects router, tool, sandbox, exception, and model events.
- `failure_context_view` returns the first exception, preceding relevant events,
  and its root-to-leaf span path.
- `tool_chain_view` and `llm_call_view` inspect a selected span.

The installed CLI exposes the same operations. Run `innertrace --help` and
`innertrace <command> --help` for the exact arguments.

## Function instrumentation

Function tracing is opt-in and does not install global hooks:

```python
from innertrace import trace_block, trace_function

@trace_function(actor="calculator", capture_return=True)
def total(values):
    return sum(values)

with trace_block("prepare", actor="pipeline", payload={"items": 3}):
    result = total([1, 2, 3])
```

See `examples/function_tracing_demo.py` for sync functions, async functions,
blocks, filters, and module-level instrumentation.

## Templates

`get_template_text()` loads the three HTML viewers from both source and wheel
installations. The templates expect endpoints backed by `list_runs()` and
`timeline_view()` when used in a web application. The timeline template also
accepts embedded projection data for the generated offline demo; normal endpoint
behavior is unchanged.

## Optional synthesis

Synthesis is not required for trace capture or deterministic projections. Copy
`innertrace/tracing/config.json.example` to `.innertrace/config.json`, then run:

```bash
innertrace timeline --last --synthesize --quality low
```

The included `low` example uses only the offline truncation provider. External
providers are optional and belong in user-owned configuration, never committed
credentials. `INNERTRACE_CONFIG_PATH` and `INNERTRACE_ENV_PATH` can point to
other files. Install `.[synthesis]` only when an HTTP synthesis provider is
needed. Without a configuration file, `--synthesize` uses deterministic offline
truncation. See the [0.3.1 synthesis migration](synthesis-migration.md) for
precedence, compatibility, and deprecation details.

## Storytelling layer

The stable `story.phase.*`, `story.objective`, `story.task`, and `story.link`
events add a semantic execution narrative without copying the underlying trace
content. `story.link` connects semantic records to execution spans. The domain
models, extractor, manager, projection, and frontend adapter remain under
`innertrace.utils.storytelling` for compatibility.
