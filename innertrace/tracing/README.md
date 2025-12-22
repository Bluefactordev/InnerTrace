# InnerTrace v0.1: Event-Sourcing Tracing for LLM Orchestration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/bluefactor/InnerTrace)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Structured tracing system for LLM orchestration with causal graph, deterministic projections, and **semantic storytelling layer**.

## Overview

This is a **systems observability project** focused on:
- Reconstructing the real execution flow (causality, branching, retry)
- Structured recording of LLM calls, tool executions, router decisions, sandbox runs
- **Semantic storytelling layer** for high-level narrative tracking (v0.1+)
- Generating standard "views" (projections) for reliable analysis

**Not about**: evaluation quality, fine-tuning, RAG quality, or custom evaluators.

## What's New in v0.1

- ✅ **Storytelling semantic layer** (`story.*` events)
- ✅ **Unified event storage** - storytelling integrates with tracer's events.jsonl
- ✅ **Projection layer** - Neutral domain model + frontend adapter separation
- ✅ **Backward compatibility** - Falls back to legacy JSON files
- ✅ **Blob-referenced content** - Large prompts/responses use blob store (no duplication)
- ✅ **Linking not copying** - story.link references execution events via span_id

## Stability Guarantees (v0.1)

### Stable API

**Event vocabulary (STABLE - guaranteed across v0.1.x):**
- Core execution events: `run.*`, `span.*`, `llm.call.*`, `tool.call.*`, `router.decision`, `sandbox.exec.*`, `exception`
- Storytelling events (stable subset):
  - `story.phase.start`, `story.phase.end` - Phase transitions
  - `story.objective` - Strategic objectives
  - `story.task` - Task events (planned, completed)
  - `story.link` - Link storytelling to execution events (via `target_span_id`)

**Storage format (STABLE):**
- `events.jsonl` - JSONL append-only event log
- `blobs/sha256/` - Content-addressable blob store
- Event schema with `ts_iso`, `type`, `run_id`, `span_id`, `payload`, etc.

**CLI commands (STABLE):**
- `./trace ls-runs` - List recent runs
- `./trace timeline <run_id>` - View timeline (canonical and compact views)
- `./trace view run <run_id>` - View compact run
- `./trace view failure <run_id>` - View failure context
- `./trace blob <ref>` - View blob content

### Experimental Features

**⚠️ EXPERIMENTAL (may change or be removed in future versions):**
- `story.quality.task`, `story.quality.phase` - Quality metrics events
- `--synthesize` flag - LLM-based synthesis of prompts/responses
- `--quality=high/low` parameter - Model selection for synthesis
- LLM synthesis configuration via `config.json`

**Note**: Experimental features are functional but may have breaking changes in minor versions (0.1.x → 0.2.0). Use with caution in production.

## Architecture

### Core Components

1. **Event Model** (`tracer.py`): JSONL append-only event log with closed vocabulary
2. **Blob Store** (`blob_store.py`): Content-addressable storage for large payloads
3. **Projections** (`projections.py`): Deterministic views for analysis
4. **CLI** (`cli.py`): Command-line interface for trace inspection

### Event Types (Closed Vocabulary)

**Core execution events:**
- `run.start`, `run.end`
- `span.start`, `span.end`
- `llm.call.start`, `llm.call.end`
- `tool.call.start`, `tool.call.end`
- `router.decision`
- `sandbox.exec.start`, `sandbox.exec.end`
- `exception`

**Storytelling events (v0.1+):**
- **STABLE**: `story.phase.start`, `story.phase.end`, `story.objective`, `story.task`, `story.link`
- **EXPERIMENTAL**: `story.quality.task`, `story.quality.phase`

## Usage

### 1. Start a Traced Run

```python
from tracing import get_tracer

tracer = get_tracer()

# Start run
run_id = tracer.start_run(
    entrypoint="api.chat",
    args={"query": "What is 2+2?", "user_id": "user123"},
    env={"api_key": "secret"}  # Will be redacted
)

try:
    # ... your application logic ...
    tracer.end_run(status="ok")
except Exception:
    tracer.end_run(status="error")
```

### 2. Create Spans

```python
# Span tracks a logical operation
with tracer.span(name="process_query", actor="agent.planner", kind="agent", tags=["planning"]) as span_id:
    # ... do work ...
    pass
```

### 3. Trace LLM Calls

```python
from tracing.tracer import emit_llm_call_start, emit_llm_call_end

# Before LLM call
emit_llm_call_start(
    tracer,
    model="gpt-4",
    prompt="What is 2+2?",
    params={"temperature": 0.7},
    purpose="reasoning"
)

# Make LLM call
response = await llm.ainvoke(messages)

# After LLM call
emit_llm_call_end(
    tracer,
    response=response.content,
    usage={"input_tokens": 100, "output_tokens": 50},
    finish_reason="stop"
)
```

### 4. Trace Tool Calls

```python
from tracing.tracer import emit_tool_call_start, emit_tool_call_end

tool_name = "web_search"
args = {"query": "python asyncio"}

emit_tool_call_start(tracer, tool_name, args)

start_time = time.time()
try:
    result = await execute_tool(tool_name, args)
    status = "ok"
    error = None
except Exception as e:
    result = None
    status = "error"
    error = str(e)
    raise
finally:
    latency_ms = int((time.time() - start_time) * 1000)
    emit_tool_call_end(tracer, tool_name, result, status, latency_ms, error)
```

### 5. Trace Router Decisions

```python
from tracing.tracer import emit_router_decision

emit_router_decision(
    tracer,
    rule="complexity_based",
    candidates=["simple", "complex"],
    chosen="complex",
    why="Query requires multi-step reasoning"
)
```

### 6. Trace Sandbox Execution

```python
from tracing.tracer import emit_sandbox_exec_start, emit_sandbox_exec_end

code = "print('hello world')"
sandbox_info = {"kind": "python", "image": "python:3.11"}

emit_sandbox_exec_start(tracer, sandbox_info, code, inputs=None)

try:
    result = await sandbox.execute(code)
    status = "ok"
    stdout = result.get("stdout")
    stderr = result.get("stderr")
except Exception as e:
    status = "error"
    stderr = str(e)
finally:
    emit_sandbox_exec_end(tracer, status, stdout, stderr, result)
```

### 7. Storytelling Layer (v0.1+)

The storytelling layer tracks high-level narrative using the **linking pattern** (no content duplication):

```python
from tracing import get_tracer, emit_llm_call_start, emit_llm_call_end
from utils.storytelling import init_storytelling

# Initialize with tracer
tracer = get_tracer()
story_manager = init_storytelling(conversation_id="conv-123", tracer=tracer)

# CORRECT v0.1 pattern: First emit llm.call via tracer, then link via story.link
with tracer.span(name="meta_planning", actor="meta_planner", kind="agent") as meta_span:
    # Emit llm.call.start and llm.call.end (creates span_id)
    emit_llm_call_start(
        tracer,
        model="gpt-4",
        prompt="Create strategic plan for document analysis",
        purpose="planning"
    )

    # ... make actual LLM call ...
    response = "Strategic objectives: 1) Extract metadata..."

    llm_span_id = emit_llm_call_end(
        tracer,
        response=response,
        usage={"input_tokens": 100, "output_tokens": 200}
    )

    # Now link storytelling context to the llm.call span (NO content duplication)
    story_manager.log_llm_call(
        phase="meta_planning",
        node="meta_planner",
        prompt="",  # Legacy parameter, not used in v0.1 events
        response="",  # Legacy parameter, not used in v0.1 events
        model_id="gpt-4",
        success=True,
        span_id=llm_span_id  # Links to execution event via span_id
    )

# EXPERIMENTAL: Log task quality metrics
story_manager.log_task_quality(
    task_id="task_01",
    quality_score=0.85,
    result_type="data_extraction",
    key_insights=["Found 100 documents", "High accuracy"],
    success_criteria_met=["All files processed"],
    execution_difficulty="medium"
)
```

**Frontend Integration**: Uses two-layer architecture (domain + adapter):

```python
from utils.storytelling.story_projection import StoryProjector
from utils.storytelling.frontend_adapter import FrontendAdapter

# Layer 1: Project events into neutral domain model
projector = StoryProjector()
projection = projector.project("conv-123")

# Layer 2: Adapt to frontend JSON format
frontend_json = FrontendAdapter.to_frontend_json(projection)

# Or use the convenience wrapper:
from utils.storytelling import StorytellingExtractor
extractor = StorytellingExtractor()
storytelling_json = extractor.load_storytelling("conv-123")
# Returns identical format as before (backward compatible)
```

### 8. Using Integration Helpers

The `integration.py` module provides convenience wrappers:

```python
from tracing.integration import (
    traced_llm_generate,
    traced_tool_call,
    trace_router_decision,
    traced_sandbox_exec,
    start_traced_run,
    end_traced_run
)

# Traced LLM call (async wrapper)
response = await traced_llm_generate(
    llm.ainvoke,
    model_id="gpt-4",
    messages=messages,
    params={"temperature": 0.7},
    purpose="reasoning"
)

# Traced tool call (decorator)
@traced_tool_call("search_web")
async def search_web(query: str):
    return {"results": [...]}

# Router decision (direct call)
trace_router_decision(
    rule="complexity",
    candidates=["fast", "deep"],
    chosen="deep",
    why="Complex query detected"
)

# Traced sandbox execution (async wrapper)
result = await traced_sandbox_exec(
    sandbox.execute,
    code="print('hello')",
    sandbox_info={"kind": "python"}
)
```

## CLI Usage

### List Recent Runs

```bash
./trace ls-runs --limit 10
```

### View Timeline (Canonical View)

**Canonical view (default) - Complete hierarchical trace:**
```bash
./trace timeline --last
```

This is the **source of truth view** - deterministic, complete, suitable for both human analysis and LLM-based debugging. Preserves full causal structure with span hierarchy.

**Compact view (lossy projection) - LLM calls only:**
```bash
./trace timeline --last --compact
```

Shows only LLM calls with component context. Useful for quick cost/performance overview, but **not suitable for root-cause analysis**.

**⚠️ EXPERIMENTAL: With LLM synthesis (summarize prompts/responses):**
```bash
./trace timeline --last --synthesize --quality=high  # High quality (default)
./trace timeline --last --synthesize --quality=low    # Low quality (faster/cheaper)
```
**Note**: The `--synthesize` and `--quality` flags are experimental and may change in future versions.

**Other filters:**
```bash
./trace timeline --last-error              # Most recent error
./trace timeline --last-ok                 # Most recent success
./trace timeline --last --endpoint api.chat_v2  # Filter by endpoint
./trace timeline --run-id <run_id>         # Explicit run ID
```

**Output format:**
- Complete hierarchy with indentation
- Component context for LLM calls: `[meta_planner]`, `[code_orchestrator #2]`
- Token count and duration on `llm.call.end`: `(7,295 tokens, 12.73s) [meta_planner]`
- Semantic status on `span.end`: `(ok, 12.87s)` (aggregated, not per-call)

### ⚠️ EXPERIMENTAL: LLM Synthesis Configuration

**Note**: This feature is experimental and may change in future versions.

When using `--synthesize`, the system uses LLM models to summarize prompts and responses. You can configure the models via `tracing/config.json` and API keys via `tracing/.env`.

**1. Configure models in `tracing/config.json`:**

```json
{
  "models": {
    "low": {
      "model": "vllm/google/gemma-3-270m-it",
      "description": "Fast and cost-effective Gemma model for quick synthesis"
    },
    "high": {
      "model": "vllm/qwen3-30b-a3b-thinking-2507-awq-4bit",
      "description": "More capable Qwen model for better quality synthesis"
    }
  },
  "default_quality": "high",
  "base_url": "http://localhost:8000/v1"
}
```

**2. Configure API key in `tracing/.env`:**

```bash
# Copy tracing/env.example to tracing/.env and fill in your OpenAI-compatible API key
BF_TRACE_SYNTHESIS_API_KEY=sk-your-api-key-here

# Optional: Override base URL (default is read from config.json)
# BF_TRACE_SYNTHESIS_BASE_URL=http://localhost:8000/v1
```

**3. Use quality parameter:**

```bash
# High quality (uses model from config.json "high" section)
./trace timeline --last --synthesize --quality=high

# Low quality (uses model from config.json "low" section)
./trace timeline --last --synthesize --quality=low
```

The `--quality` parameter selects which model to use from `config.json`. This allows you to choose between faster/cheaper models (low) or more capable models (high) based on your needs.

### View Compact Run

**With run ID:**
```bash
./trace view run --run-id <run_id>
```

**Most recent run:**
```bash
./trace view run --last
```

**Most recent error:**
```bash
./trace view run --last-error
```

### View Failure Context

**With run ID:**
```bash
./trace view failure --run-id <run_id> --n 80
```

**Most recent error (default):**
```bash
./trace view failure --last
```

**Most recent successful run:**
```bash
./trace view failure --last-ok
```

### View Tool Chain

```bash
python -m tracing.cli view tool-chain --span-id <span_id>
```

### View LLM Call Details

```bash
python -m tracing.cli view llm --span-id <span_id>
```

### View Blob Content

```bash
python -m tracing.cli blob --ref blob:sha256:<hash>
```

## Projections

Projections are deterministic views that reduce complexity for analysis:

### 1. compact_run_view

Returns a compact list of key events (router decisions, tool calls, LLM calls, exceptions).

```python
from tracing.projections import compact_run_view

view = compact_run_view(run_id)
# Returns: {"run_id": "...", "items": [...]}
```

### 2. failure_context_view

Returns context around first exception (last N events + parent span tree).

```python
from tracing.projections import failure_context_view

view = failure_context_view(run_id, n=80)
# Returns: {"run_id": "...", "exception": {...}, "preceding_events": [...], "span_tree": [...]}
```

### 3. tool_chain_view

Returns all tool calls descending from a span.

```python
from tracing.projections import tool_chain_view

view = tool_chain_view(span_id)
# Returns: {"span_id": "...", "tool_calls": [...]}
```

### 4. llm_call_view

Returns detailed LLM call info (refs + previews, not full content).

```python
from tracing.projections import llm_call_view

view = llm_call_view(span_id)
# Returns: {"span_id": "...", "model": "...", "prompt_ref": "...", "usage": {...}, ...}
```

## Integration Points

To integrate tracing into your codebase, add hooks at these central points:

### 1. ModelProvider.generate / generate_simple

Add tracing around LLM calls in the generate methods.

**Location**: `utils/models/model_provider.py`

**Before**:
```python
response = await llm_instance.ainvoke(messages, **kwargs)
```

**After**:
```python
from tracing.integration import traced_llm_generate

response = await traced_llm_generate(
    llm_instance.ainvoke,
    model_id=model_id,
    messages=messages,
    params={"temperature": kwargs.get("temperature", 0.7)},
    purpose="generation"
)
```

### 2. Tool Dispatcher / Registry

Add tracing around tool execution.

**Location**: Find central tool execution point (likely in tool handling code)

**Pattern**:
```python
from tracing.integration import traced_tool_call

@traced_tool_call("tool_name")
async def execute_tool(tool_name: str, args: dict):
    # ... existing tool execution logic ...
    pass
```

### 3. Router Decision Function

Add router decision logging.

**Location**: Find where routing decisions are made

**Pattern**:
```python
from tracing.integration import trace_router_decision

# After making routing decision
trace_router_decision(
    rule="rule_name",
    candidates=["option1", "option2"],
    chosen=selected_option,
    why="reasoning for decision"
)
```

### 4. Sandbox Execution Entrypoint

Add tracing around sandbox execution.

**Location**: Find sandbox execution entrypoint

**Pattern**:
```python
from tracing.integration import traced_sandbox_exec

result = await traced_sandbox_exec(
    sandbox.execute,
    code=code,
    sandbox_info={"kind": "python", "image": "python:3.11"},
    inputs=inputs
)
```

### 5. Global Exception Boundary

Add exception tracing at top-level error handlers.

**Pattern**:
```python
from tracing import get_tracer

try:
    # ... application logic ...
except Exception as e:
    tracer = get_tracer()
    tracer.emit_exception(e, actor="app.main")
    raise
```

## Storage Structure

```
traces/
├── events.jsonl          # Append-only event log
└── blobs/
    └── sha256/
        ├── <hash>.txt    # Text blobs (prompts, code, stdout, etc.)
        └── <hash>.json   # JSON blobs (structured data)
```

## Invariants

1. Every run has exactly 1 `run.start` and 1 `run.end`
2. Every span has exactly 1 `span.start` and 1 `span.end`
3. Every `llm.call.start` has corresponding `llm.call.end` in same span
4. Every `tool.call.start` has corresponding `tool.call.end` in same span
5. No secrets in cleartext (automatic redaction)

## Secret Redaction

Automatic redaction of sensitive keys:
- `api_key`, `token`, `password`, `secret`, `authorization`, etc.
- Pattern matching for Bearer tokens, API keys, etc.

## Dependencies

Minimal dependencies (stdlib only):
- `json`, `hashlib`, `pathlib`, `threading`, `contextvars`, `time`, `traceback`

Optional:
- `ulid` for better run IDs (falls back to `uuid`)

## Example: Full Integration

```python
from tracing import get_tracer
from tracing.integration import start_traced_run, end_traced_run

# Application entry point
def main():
    run_id = start_traced_run(
        entrypoint="cli.main",
        args={"command": "chat", "query": "hello"}
    )

    start_time = time.time()

    try:
        # Get tracer
        tracer = get_tracer()

        # Create main span
        with tracer.span("main_execution", "app", "function"):
            # ... your application logic with nested spans ...
            result = process_query(query)

        # End run successfully
        latency_ms = int((time.time() - start_time) * 1000)
        end_traced_run("ok", latency_ms)

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        end_traced_run("error", latency_ms)
        raise
```

## Performance Considerations

- Events are written append-only (fast)
- Blob storage is idempotent (deduplication via content-addressing)
- Thread-safe and exception-safe (won't break runs)
- Automatic secret redaction
- Preview truncation for large content

## Next Steps

1. Add hooks to ModelProvider LLM calls
2. Add hooks to tool execution dispatcher
3. Add hooks to router decision points
4. Add hooks to sandbox execution
5. Add exception boundary at API entry points
6. Test with real runs
7. Use CLI to inspect traces
8. Analyze with projections
