# Synthesis configuration migration in 0.3.1

Trace capture, `events.jsonl`, blob storage, and the default timeline do not
depend on synthesis configuration. This migration affects only
`innertrace timeline --synthesize`.

## Before and after

| Behavior | Before 0.3.1 | 0.3.1 |
| --- | --- | --- |
| Preferred config | `innertrace/tracing/config.json` beside package code | `.innertrace/config.json` in the working directory |
| Preferred env file | `innertrace/tracing/.env` beside package code | `.innertrace/.env` in the working directory |
| Explicit paths | Not the documented primary path | `INNERTRACE_CONFIG_PATH` and `INNERTRACE_ENV_PATH` |
| No config | `--synthesize` exited with an error | deterministic offline truncation |
| Packaged defaults | Could inherit deployment-specific files | only neutral `.example` files are included |

Resolution order is:

1. the explicit `INNERTRACE_CONFIG_PATH` or `INNERTRACE_ENV_PATH` value;
2. the matching file under `.innertrace/` in the current working directory;
3. the old package-local file, if it exists, as a deprecated compatibility
   fallback.

An explicit path is authoritative. If it does not exist or cannot be read,
InnerTrace does not silently fall through to another configuration. Process
environment variables also continue to take precedence over values loaded from
an env file.

## Migrate an existing installation

Move, rather than copy, deployment-owned files out of the installed package:

```bash
mkdir -p .innertrace
mv path/to/innertrace/tracing/config.json .innertrace/config.json
mv path/to/innertrace/tracing/.env .innertrace/.env  # if present
```

The package-local fallback remains available in 0.3.1 so existing source-based
integrations keep working. The CLI prints a deprecation warning when it uses
that fallback. It is not included in wheel or sdist defaults and may be removed
in a future minor release.

The tracked `config.json.example` is deliberately neutral. Its `low` quality
uses only the offline truncation provider; external endpoints, model names, and
credentials must remain in user-owned configuration.
