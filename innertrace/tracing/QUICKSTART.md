# Quick Start Guide - LLM Execution Observability

Get started with the tracing system in 5 minutes.

## 1. Run the Example

```bash
# Run the demonstration
python3 -m tracing.example
```

This creates two example traces:
1. A successful workflow with LLM calls, tools, and sandbox execution
2. A failure scenario with exception tracing

## 2. List Runs

```bash
# List recent runs
./trace ls-runs
```

Output:
```
Run ID                                   Entrypoint           Status  Start Time
---------------------------------------------------------------------------------------------
e09155d7-37c4-4612-9ec5-4414f2312eea     example.error        error   2025-12-17 13:35:48
b56e211b-140b-411d-a3ba-5c7d004a9a1a     example.workflow     ok      2025-12-17 13:35:46
```

## 3. View Timeline (Recommended for Debug)

```bash
# View timeline of most recent run (no need to know run_id!)
./trace timeline --last

# View timeline of most recent error
./trace timeline --last-error

# View timeline of most recent success
./trace timeline --last-ok

# With explicit run_id (if you have it)
./trace timeline --run-id b56e211b-140b-411d-a3ba-5c7d004a9a1a
```

This shows:
- **Run ID**: Full run ID on the first line (no truncation)
- Human-readable timestamps (HH:MM:SS.sss)
- Delta times between events (+XXXms)
- Span hierarchy with indentation (tool calls nested under the LLM span that requested them)
- Key payload details for each event
- **Tool results/errors**: For each tool call end, a short preview of the result or error (what the model received). With `--synthesize`, tool results are summarized (same provider as prompt/response; essential, concise).
- **Exceptions**: Full error message for exception events

**💡 Tip**: Use `--last` to debug immediately without looking up run IDs!

## 4. View a Run

```bash
# View compact run (most recent)
./trace view run --last

# View compact run (most recent error)
./trace view run --last-error

# With explicit run_id
./trace view run --run-id b56e211b-140b-411d-a3ba-5c7d004a9a1a
```

This shows:
- Router decisions
- LLM calls (with blob references)
- Tool executions
- Sandbox runs
- Exceptions

## 5. View Failure Context

```bash
# View failure context (most recent error - default)
./trace view failure --last

# With explicit run_id
./trace view failure --run-id e09155d7-37c4-4612-9ec5-4414f2312eea
```

This shows:
- Exception details (type, message, location)
- Last 80 events before exception
- Parent span tree (root to exception)

## 6. View Blob Content

```bash
# Get a blob reference from a view command, then:
./trace blob --ref blob:sha256:525973d6ede3a28740ea20349b4df59f462540d869969bea98a0e76aa42b3a2a
```

This shows the actual content (prompt, response, code, etc.)

## 7. Integrate into Your Code

### Minimal Integration (API endpoint)

```python
from tracing.integration import start_traced_run, end_traced_run
import time

@app.route('/api/chat', methods=['POST'])
async def chat():
    # Start traced run
    run_id = start_traced_run("api.chat", {"query": request.json.get("query")})
    start_time = time.time()

    try:
        # Your existing code here
        result = await process_chat(request.json)

        # End successfully
        latency_ms = int((time.time() - start_time) * 1000)
        end_traced_run("ok", latency_ms)
        return jsonify(result)

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        end_traced_run("error", latency_ms)
        raise
```

### Add LLM Call Tracing

```python
from tracing import get_tracer
from tracing.tracer import emit_llm_call_start, emit_llm_call_end

# Before LLM call
tracer = get_tracer()
if tracer.current_run_id():
    emit_llm_call_start(tracer, model="gpt-4", prompt=messages, params={"temperature": 0.7})

# Make call
response = await llm.ainvoke(messages)

# After LLM call
if tracer.current_run_id():
    usage = {"input_tokens": 100, "output_tokens": 50}  # Extract from response
    emit_llm_call_end(tracer, response, usage)
```

### Add Router Decision Tracing

```python
from tracing.integration import trace_router_decision

# After making routing decision
trace_router_decision(
    rule="depth_based",
    candidates=["simple", "complex"],
    chosen="complex",
    why="Query requires multi-step reasoning"
)
```

## 8. Analyze Traces Programmatically

```python
from tracing.projections import compact_run_view, failure_context_view

# Get compact view of a run
view = compact_run_view(run_id="...")
for item in view["items"]:
    if item["type"] == "exception":
        print(f"Exception: {item['exc_type']} - {item['message']}")

# Get failure context
failure = failure_context_view(run_id="...")
print(f"Exception at: {failure['exception']['where']}")
print(f"Last {len(failure['preceding_events'])} events before failure")
```

## File Structure

After running the example:

```
traces/
├── events.jsonl              # 30 events from 2 runs
└── blobs/
    └── sha256/
        ├── <hash>.txt        # Text content (prompts, responses, code)
        └── <hash>.json       # JSON content (args, results)
```

## What Gets Traced

✅ Run lifecycle (start/end with status)
✅ Spans (nested execution contexts)
✅ LLM calls (prompts, responses, token usage)
✅ Tool calls (args, results, timing)
✅ Router decisions (candidates, chosen, reasoning)
✅ Sandbox execution (code, stdout/stderr, results)
✅ Exceptions (type, message, location, stack trace)

## CLI Commands Cheat Sheet

```bash
# List runs
./trace ls-runs [--limit 20]

# Timeline (recommended for debug)
./trace timeline --last              # Most recent run
./trace timeline --last-error        # Most recent error
./trace timeline --last-ok           # Most recent success
./trace timeline --run-id <run_id>   # With explicit ID

# View run
./trace view run --last              # Most recent
./trace view run --last-error        # Most recent error
./trace view run --run-id <run_id>   # With explicit ID

# View failure
./trace view failure --last          # Most recent error (default)
./trace view failure --run-id <run_id> [--n 80]

# View tool chain
./trace view tool-chain --span-id <span_id>

# View LLM call
./trace view llm --span-id <span_id>

# View blob
./trace blob --ref blob:sha256:<hash>
```

## Python API Cheat Sheet

```python
from tracing import get_tracer
from tracing.integration import start_traced_run, end_traced_run, trace_router_decision
from tracing.tracer import (
    emit_llm_call_start, emit_llm_call_end,
    emit_tool_call_start, emit_tool_call_end,
    emit_sandbox_exec_start, emit_sandbox_exec_end
)

# Start/end run
run_id = start_traced_run("entrypoint", {"arg": "value"})
end_traced_run("ok", latency_ms=1000)

# Create span
tracer = get_tracer()
with tracer.span("operation_name", "actor", "kind"):
    # ... do work ...
    pass

# Emit events (if inside a traced run)
if tracer.current_run_id():
    emit_llm_call_start(tracer, model, prompt, params)
    emit_llm_call_end(tracer, response, usage)
    emit_tool_call_start(tracer, tool_name, args)
    emit_tool_call_end(tracer, tool_name, result, "ok", latency_ms)
    trace_router_decision(rule, candidates, chosen, why)
    # ... etc
```

## Next Steps

1. ✅ Run example: `python3 -m tracing.example`
2. ✅ Explore CLI: `./trace ls-runs`
3. 📖 Read full guide: [README.md](README.md)
4. 🔧 Integration guide: [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
5. 🚀 Add to your API endpoint
6. 📊 Analyze your first traced run

## Support

- Full documentation: [README.md](README.md)
- Integration guide: [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- Example code: [example.py](example.py)
- Implementation summary: [../TRACING_IMPLEMENTATION.md](../TRACING_IMPLEMENTATION.md)

## Remember

This is a **systems observability project**, not about:
- ❌ Evaluation quality
- ❌ Fine-tuning or RAG
- ❌ Model performance metrics

It's about:
- ✅ Reconstructing execution flow
- ✅ Structured event recording
- ✅ Causal debugging
- ✅ Deterministic analysis
