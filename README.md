# Debug the agent run, not just the final error

InnerTrace reconstructs the decisions, tool calls, causal branches, retries, and
exceptions that led an LLM or agent to its result—from a local append-only trace.

[![CI](https://github.com/Bluefactordev/InnerTrace/actions/workflows/ci.yml/badge.svg)](https://github.com/Bluefactordev/InnerTrace/actions/workflows/ci.yml)
[![Python 3.9–3.12](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Bluefactordev/InnerTrace/blob/codex/innertrace-star-readiness/LICENSE)

![A real InnerTrace timeline showing a stale-cache branch, its exception, retry, and corrected result](https://raw.githubusercontent.com/Bluefactordev/InnerTrace/616d0f3b6ed31d8a5bbfabdac30d9304dc2a7f88/docs/assets/agent-failure-demo.png)

In this demo an inventory agent first answers from stale data. InnerTrace shows
why the router chose that source, the tool result it received, the failing
branch, the retry, and the corrected final answer.

## Quick Start

Install InnerTrace 0.3.1 and run its offline demo:

```bash
python -m pip install innertrace==0.3.1
innertrace demo
innertrace --events-path demo_output/traces/events.jsonl timeline --last
```

The demo scenario and final result are deterministic, offline, and need no API
key. Run IDs and timestamps are generated afresh. It writes a real
`events.jsonl`, content-addressed blobs, and a self-contained HTML timeline.
Expect output shaped like this:

```text
Run ID: <generated-run-id>
Trace: .../demo_output/traces/events.jsonl
HTML demo: .../demo_output/agent-failure.html
Final result: CHAIR-42 has 7 units (source: inventory.primary)
```

Read the [usage and integration guide](https://github.com/Bluefactordev/InnerTrace/blob/codex/innertrace-star-readiness/docs/guide.md), the
[event and storage contract](https://github.com/Bluefactordev/InnerTrace/blob/codex/innertrace-star-readiness/docs/event-format.md), or the
[0.3 changelog](https://github.com/Bluefactordev/InnerTrace/blob/codex/innertrace-star-readiness/CHANGELOG.md).

## When InnerTrace is useful

Use InnerTrace when the final agent response does not explain what actually
happened: which route was selected, which tool returned a misleading value,
where an exception occurred, or whether a retry changed the outcome. It is a
good fit for local development, reproducible bug reports, incident analysis,
and applications that need filesystem-owned trace data.

It is not an evaluator, prompt optimizer, or hosted monitoring service. It does
not decide whether an answer is good; it records execution facts and derives
read-only views from them.

## How it differs

- **Application logs** remain useful for free-form messages and operational
  context. InnerTrace adds run/span causality plus agent-specific events and can
  be used alongside normal logging.
- **OpenTelemetry** is a broad telemetry standard and ecosystem. InnerTrace is
  a focused local JSONL/blob format with deterministic agent-run projections;
  it is not a collector, exporter, or universal replacement for OpenTelemetry.
- **Hosted LLM observability platforms** may provide managed dashboards,
  collaboration, evaluation, and retention. InnerTrace keeps its core path
  local and account-free. Choose or combine them according to your operational
  needs.

## Add tracing to an agent

All imports below are part of the installed `innertrace` package:

```python
from innertrace import Tracer
from innertrace.tracing import (
    emit_router_decision,
    emit_tool_call_end,
    emit_tool_call_start,
)

tracer = Tracer(
    events_path="traces/events.jsonl",
    blobs_path="traces/blobs",
)
run_id = tracer.start_run("assistant.answer", {"request": "stock for CHAIR-42"})

with tracer.span("answer", actor="agent.inventory", kind="agent"):
    emit_router_decision(
        tracer,
        rule="freshness",
        candidates=["cache", "primary"],
        chosen="primary",
    )
    with tracer.span("lookup", actor="tool.inventory", kind="tool"):
        emit_tool_call_start(tracer, "inventory.lookup", {"sku": "CHAIR-42"})
        emit_tool_call_end(
            tracer,
            "inventory.lookup",
            {"available": 7},
            status="ok",
            latency_ms=4,
        )

tracer.end_run("ok")
print(run_id)
```

Tracing is inert outside an active run. The integration wrappers under
`innertrace.tracing.integration` can instrument existing async LLM, tool,
router, and sandbox calls without changing the event vocabulary.

## Inspect traces

The installed CLI reads traces without mutating them:

```bash
innertrace ls-runs
innertrace timeline --last
innertrace view run --last
innertrace view failure --run-id <run-id>
innertrace view tool-chain --span-id <span-id>
innertrace blob --ref blob:sha256:<hash>
```

Global paths precede the subcommand:

```bash
innertrace \
  --events-path /path/to/events.jsonl \
  --blobs-path /path/to/blobs \
  timeline --last
```

The canonical timeline preserves the span hierarchy. `--compact` is a lossy
overview intended for quick inspection, not root-cause analysis.

## What is recorded

Each line of `events.jsonl` contains timestamps, run/span IDs, parent causality,
actor, level, tags, event type, and a redacted payload. Large arguments,
results, prompts, responses, code, and stack traces live in
`blobs/sha256/`; events keep their `blob:sha256:...` references.

The established vocabulary includes:

| Layer | Events |
| --- | --- |
| Lifecycle | `run.start`, `run.end`, `span.start`, `span.end`, `exception` |
| Models | `llm.call.start`, `llm.call.end` |
| Tools and routing | `tool.call.start`, `tool.call.end`, `router.decision` |
| Sandboxes | `sandbox.exec.start`, `sandbox.exec.end`, `code_execution_error` |
| Story | `story.phase.start`, `story.phase.end`, `story.objective`, `story.task`, `story.link` |

InnerTrace 0.3.1 preserves the 0.2 public imports, event names, JSONL records,
blob references, projections, function tracing, and storytelling APIs. The
`story.quality.*` events and optional synthesis settings remain experimental.

## HTML viewers

The wheel contains the existing index, runs, and timeline templates. Embedding
applications can load them without relying on a checkout path:

```python
from innertrace import get_template_text, template_names

print(template_names())
timeline_html = get_template_text("timeline.html")
```

The committed screenshot is generated from the real offline demo and bundled
timeline template:

```bash
python examples/agent_failure_demo.py \
  --output-dir .demo-output \
  --html docs/demo/agent-failure.html \
  --screenshot docs/assets/agent-failure-demo.png
```

Refreshing the PNG requires a local Chrome or Chromium executable. Generating
the trace and HTML does not.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m build
python -m twine check dist/*
```

See [CONTRIBUTING.md](https://github.com/Bluefactordev/InnerTrace/blob/codex/innertrace-star-readiness/CONTRIBUTING.md)
before changing event or storage contracts. Release publication and signing
remain explicit release-gate actions.
