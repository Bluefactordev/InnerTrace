# Contributing

InnerTrace favors small, deterministic instrumentation and projections that do
not change host-application behavior.

## Set up a development checkout

```bash
git clone https://github.com/Bluefactordev/InnerTrace.git
cd InnerTrace
python -m pip install -e '.[dev]'
python -m pytest
```

Before submitting a change, also run:

```bash
python -m build
python -m twine check dist/*
innertrace --help
python examples/agent_failure_demo.py
```

Keep changes focused. Event emitters must preserve existing event names and
payload keys, and projections must treat `events.jsonl` as append-only source
data. Add regression tests when touching redaction, serialization, storage,
context propagation, CLI behavior, package resources, or storytelling links.

Never commit generated local traces, blobs, credentials, `.env` files, or
provider-specific synthesis configuration. Explain user-visible behavior and
compatibility impact in the pull request.
