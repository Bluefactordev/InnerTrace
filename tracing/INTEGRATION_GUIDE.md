# Integration Guide: Adding Tracing to Existing Codebase

This guide shows how to add minimal tracing hooks to your existing codebase without extensive refactoring.

## Philosophy

- **Minimal invasiveness**: Add hooks only at central points
- **No refactoring**: Don't restructure existing code
- **Safe by default**: Tracing failures won't break runs
- **Additive only**: Add new code, don't modify core logic

## Integration Points

### 1. ModelProvider LLM Calls

**Location**: `utils/models/model_provider.py`

**Goal**: Trace all LLM API calls with prompts, responses, token usage

#### Option A: Wrapper at generate_simple (Recommended)

Add tracing at the beginning and end of `generate_simple` method:

```python
# At the top of generate_simple (around line 2510)
async def generate_simple(
    model_id: str,
    messages: List[Dict[str, str]],
    tools: Optional[List[str]] = None,
    response_schema: Optional[Type[BaseModel]] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Union[str, Dict[str, Any], BaseModel]:

    # ADD: Import tracing
    from tracing import get_tracer
    from tracing.tracer import emit_llm_call_start, emit_llm_call_end

    # ADD: Start tracing if we have an active run
    tracer = get_tracer()
    if tracer.current_run_id():
        # Convert messages to string for tracing
        prompt_text = "\n".join([
            f"{m.get('role', 'unknown')}: {m.get('content', '')[:200]}"
            for m in messages[:3]  # First 3 messages
        ])

        emit_llm_call_start(
            tracer,
            model=model_id,
            prompt=messages,  # Full messages will be stored in blob
            params={"tools": tools, **kwargs},
            purpose="generate_simple"
        )

    # ... existing code ...

    # Before returning (around line 3200+)
    # ADD: End tracing
    if tracer.current_run_id():
        usage = None
        if hasattr(response, 'usage_metadata'):
            usage = {
                "input_tokens": getattr(response.usage_metadata, 'input_tokens', 0),
                "output_tokens": getattr(response.usage_metadata, 'output_tokens', 0),
            }

        emit_llm_call_end(
            tracer,
            response=response,
            usage=usage
        )

    return response
```

#### Option B: Wrapper at ainvoke calls

Create a wrapper function and replace direct `ainvoke` calls:

```python
# Add this helper function to ModelProvider class
async def _traced_ainvoke(self, llm, messages, model_id, **kwargs):
    """Wrapper for llm.ainvoke with tracing."""
    from tracing import get_tracer
    from tracing.tracer import emit_llm_call_start, emit_llm_call_end

    tracer = get_tracer()

    if tracer.current_run_id():
        emit_llm_call_start(tracer, model_id, messages, kwargs)

    response = await llm.ainvoke(messages, **kwargs)

    if tracer.current_run_id():
        usage = None
        if hasattr(response, 'usage_metadata'):
            usage = {
                "input_tokens": getattr(response.usage_metadata, 'input_tokens', 0),
                "output_tokens": getattr(response.usage_metadata, 'output_tokens', 0),
            }
        emit_llm_call_end(tracer, response, usage)

    return response

# Then replace calls like:
# response = await llm_instance.ainvoke(messages, **kwargs)
# With:
# response = await self._traced_ainvoke(llm_instance, messages, model_id, **kwargs)
```

### 2. Tool Calls

**Location**: Find the central tool dispatcher (search for tool execution logic)

Common locations:
- `utils/tools/` directory
- Tool registry or dispatcher
- LangGraph tool nodes

#### Example: Tool Node Wrapper

```python
# In the tool execution code (e.g., utils/langgraph/nodes/code_orchestrator.py or similar)

from tracing import get_tracer
from tracing.tracer import emit_tool_call_start, emit_tool_call_end
import time

async def execute_tool_with_tracing(tool_name: str, tool_args: dict, tool_func):
    """Wrapper for tool execution with tracing."""
    tracer = get_tracer()

    # Emit start event
    if tracer.current_run_id():
        emit_tool_call_start(tracer, tool_name, tool_args, actor="tool")

    start_time = time.time()
    status = "ok"
    result = None
    error = None

    try:
        result = await tool_func(**tool_args)
        return result
    except Exception as e:
        status = "error"
        error = str(e)
        if tracer.current_run_id():
            tracer.emit_exception(e, actor="tool")
        raise
    finally:
        if tracer.current_run_id():
            latency_ms = int((time.time() - start_time) * 1000)
            emit_tool_call_end(tracer, tool_name, result, status, latency_ms, error)

# Use in tool dispatcher:
# result = await execute_tool_with_tracing(tool_name, args, tool_function)
```

#### Example: Decorator Approach

If you have individual tool functions:

```python
from tracing.integration import traced_tool_call

@traced_tool_call("web_search")
async def web_search(query: str, num_results: int = 5):
    # ... existing tool implementation ...
    pass

@traced_tool_call("code_executor")
async def code_executor(code: str, language: str = "python"):
    # ... existing tool implementation ...
    pass
```

### 3. Router Decisions

**Location**: Find where routing decisions are made

Possible locations:
- Router selection logic in ModelProvider
- Quality budget selection
- Depth-based routing (simple vs planner)
- Model selection logic

#### Example: Depth-based Router

```python
# In ModelProvider.generate (around line 1953-1988)

from tracing.integration import trace_router_decision

if depth == 0:
    # ADD: Trace router decision
    if tracer.current_run_id():
        trace_router_decision(
            rule="depth_based",
            candidates=["simple_pipeline", "planner_workflow"],
            chosen="simple_pipeline",
            why=f"depth=0 with quality_budget={quality_budget}"
        )

    # Existing code: Modalità Veloce
    result = await ModelProvider.generate_simple(...)
else:
    # ADD: Trace router decision
    if tracer.current_run_id():
        trace_router_decision(
            rule="depth_based",
            candidates=["simple_pipeline", "planner_workflow"],
            chosen="planner_workflow",
            why=f"depth={depth}, requires hierarchical planning"
        )

    # Existing code: Modalità Approfondita
    result = await ModelProvider.generate_with_planner(...)
```

#### Example: Model Selection

```python
# If there's model selection logic
from tracing.integration import trace_router_decision

selected_model = select_model(query_complexity, budget)

if tracer.current_run_id():
    trace_router_decision(
        rule="model_selection",
        candidates=available_models,
        chosen=selected_model,
        why=f"complexity={query_complexity}, budget={budget}"
    )
```

### 4. Sandbox Execution

**Location**: Find code execution sandbox entry point

Possible locations:
- `utils/langgraph/nodes/code_orchestrator.py`
- Python/code execution tools
- Sandbox wrapper modules

#### Example: Code Orchestrator Node

```python
# In code_orchestrator.py or similar

from tracing import get_tracer
from tracing.tracer import emit_sandbox_exec_start, emit_sandbox_exec_end

async def execute_code_in_sandbox(code: str, language: str = "python"):
    """Execute code in sandbox with tracing."""
    tracer = get_tracer()

    sandbox_info = {
        "kind": language,
        "image": f"{language}:latest"  # Or actual image name
    }

    # Emit start event
    if tracer.current_run_id():
        emit_sandbox_exec_start(tracer, sandbox_info, code, actor="sandbox")

    status = "ok"
    stdout = None
    stderr = None
    result = None
    error = None

    try:
        # ... existing sandbox execution code ...
        result = await sandbox.execute(code, language=language)

        # Extract stdout/stderr if available
        if isinstance(result, dict):
            stdout = result.get("stdout")
            stderr = result.get("stderr")

        return result

    except Exception as e:
        status = "error"
        error = str(e)
        if tracer.current_run_id():
            tracer.emit_exception(e, actor="sandbox")
        raise

    finally:
        if tracer.current_run_id():
            emit_sandbox_exec_end(tracer, status, stdout, stderr, result, error)
```

### 5. Run Start/End at Entry Points

**Location**: API route handlers or main entry points

Possible locations:
- `routes/*.py` (Flask/FastAPI routes)
- CLI entry points
- Main application handlers

#### Example: API Route

```python
# In routes/chat.py or similar

from tracing import get_tracer
from tracing.integration import start_traced_run, end_traced_run
import time

@app.route('/api/chat', methods=['POST'])
async def chat_endpoint():
    data = request.json

    # Start traced run
    run_id = start_traced_run(
        entrypoint="api.chat",
        args={
            "query": data.get("query"),
            "user_id": g.user_id,
            # Don't include sensitive data
        }
    )

    start_time = time.time()

    try:
        # Get tracer for nested spans
        tracer = get_tracer()

        # Create main span
        with tracer.span("chat_request", "api.chat", "function"):
            # ... existing chat logic ...
            result = await process_chat(data)

        # End run successfully
        latency_ms = int((time.time() - start_time) * 1000)
        end_traced_run("ok", latency_ms)

        return jsonify(result)

    except Exception as e:
        # End run with error
        latency_ms = int((time.time() - start_time) * 1000)
        end_traced_run("error", latency_ms)
        raise
```

#### Example: CLI Entry Point

```python
# In main CLI script

from tracing.integration import start_traced_run, end_traced_run
import time
import sys

def main():
    args = parse_args()

    # Start traced run
    run_id = start_traced_run(
        entrypoint="cli.main",
        args={"command": args.command, "input": args.input}
    )

    start_time = time.time()

    try:
        # ... existing CLI logic ...
        result = execute_command(args)

        # End run successfully
        latency_ms = int((time.time() - start_time) * 1000)
        end_traced_run("ok", latency_ms)

        return 0

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        end_traced_run("error", latency_ms)
        print(f"Error: {e}", file=sys.stderr)
        return 1
```

### 6. Exception Boundaries

**Location**: Top-level error handlers

#### Example: Global Exception Handler

```python
# In Flask app configuration

from tracing import get_tracer

@app.errorhandler(Exception)
def handle_exception(e):
    tracer = get_tracer()

    # Emit exception event if we're in a traced run
    if tracer.current_run_id():
        tracer.emit_exception(e, actor="app.error_handler")

    # ... existing error handling ...
    return jsonify({"error": str(e)}), 500
```

## Integration Checklist

- [ ] Add tracing to LLM calls in ModelProvider
- [ ] Add tracing to tool execution dispatcher
- [ ] Add tracing to router decision points
- [ ] Add tracing to sandbox execution
- [ ] Add run start/end to API routes
- [ ] Add run start/end to CLI entry points
- [ ] Add exception tracing to error handlers
- [ ] Test with a real run
- [ ] Verify events are written to `traces/events.jsonl`
- [ ] Verify blobs are stored in `traces/blobs/sha256/`
- [ ] Test CLI: `./trace ls-runs`
- [ ] Test CLI: `./trace view run --run-id <id>`

## Testing Your Integration

### 1. Run a Simple Test

```python
# test_tracing.py

import asyncio
from tracing import get_tracer
from tracing.integration import start_traced_run, end_traced_run

async def test_run():
    # Start run
    run_id = start_traced_run("test.simple", {"test": "hello"})
    print(f"Started run: {run_id}")

    tracer = get_tracer()

    # Create a span
    with tracer.span("test_operation", "test", "function"):
        # Simulate work
        await asyncio.sleep(0.1)
        print("Work done")

    # End run
    end_traced_run("ok", 100)
    print(f"Run completed: {run_id}")
    print(f"Check traces/events.jsonl for events")

if __name__ == "__main__":
    asyncio.run(test_run())
```

Run it:
```bash
python test_tracing.py
```

### 2. Check Events

```bash
# View the events file
cat traces/events.jsonl | python -m json.tool

# Or use the CLI
./trace ls-runs
```

### 3. Check Blobs

```bash
# List blobs
ls -lh traces/blobs/sha256/

# View a blob
./trace blob --ref blob:sha256:<hash>
```

## Gradual Rollout

You can integrate tracing gradually:

1. **Phase 1**: Add run start/end to one API endpoint
2. **Phase 2**: Add LLM call tracing to generate_simple
3. **Phase 3**: Add tool call tracing
4. **Phase 4**: Add router decision tracing
5. **Phase 5**: Add sandbox execution tracing
6. **Phase 6**: Add exception boundaries

Each phase is independently useful and can be deployed separately.

## Conditional Tracing

You can make tracing conditional (e.g., only for certain users or requests):

```python
# Only trace if flag is set
if should_trace(request):
    run_id = start_traced_run(...)
    # ... traced logic ...
    end_traced_run("ok")
else:
    # ... normal logic without tracing ...
    pass
```

## Performance Impact

- Event writing: ~0.1-0.5ms per event (append to JSONL)
- Blob storage: ~1-5ms per blob (write to file)
- Total overhead: <1% for typical runs with 10-100 events

Tracing is:
- Exception-safe (won't crash runs)
- Thread-safe (concurrent writes OK)
- Optional (check `current_run_id()` before emitting)

## Troubleshooting

### Events not appearing

Check:
1. Is `tracer.start_run()` called?
2. Does `tracer.current_run_id()` return a value?
3. Check file permissions on `traces/` directory
4. Check for exceptions in tracing code (should be logged)

### Blobs not storing

Check:
1. File permissions on `traces/blobs/` directory
2. Disk space
3. Check for hash collisions (very rare)

### Secret leakage

Check:
1. Redaction is automatic for common secret keys
2. Add custom redaction if needed in `tracing/redact.py`
3. Review blob contents: `./trace blob --ref <ref>`

## Next Steps

1. Choose one integration point to start (recommend: API route)
2. Add minimal hooks following examples above
3. Test with a real request
4. Verify events with CLI
5. Gradually add more integration points
6. Use projections for analysis
