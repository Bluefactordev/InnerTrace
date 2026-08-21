# Event and storage contract

InnerTrace stores an append-only newline-delimited JSON stream at
`traces/events.jsonl` by default. Each event contains:

```json
{
  "ts": 1720000000.0,
  "ts_human": "12:00:00.000",
  "ts_iso": "2024-07-03T12:00:00.000Z",
  "run_id": "...",
  "span_id": "...",
  "parent_span_id": null,
  "type": "router.decision",
  "actor": "router.inventory",
  "level": "info",
  "tags": [],
  "payload": {}
}
```

`dt_ms` appears after the first event in a run. Payload keys depend on the event
type; existing keys and event names are compatibility-sensitive.

## Core vocabulary

- `run.start`, `run.end`
- `span.start`, `span.end`, `exception`
- `llm.call.start`, `llm.call.end`
- `tool.call.start`, `tool.call.end`
- `router.decision`
- `sandbox.exec.start`, `sandbox.exec.end`, `code_execution_error`

The stable storytelling subset is `story.phase.start`, `story.phase.end`,
`story.objective`, `story.task`, and `story.link`. The `story.quality.*` records
are experimental.

## Blob store

Large content is stored by SHA-256 under `blobs/sha256/`. An event references it
as `blob:sha256:<digest>`. Repeated identical content resolves to the same blob.
The CLI reads a reference with:

```bash
innertrace --blobs-path traces/blobs blob --ref blob:sha256:<digest>
```

Trace files can contain application data. Keep them out of version control,
apply access controls appropriate to the host application, and review redaction
behavior before production capture.
