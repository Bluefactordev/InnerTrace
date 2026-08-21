# Changelog

All notable InnerTrace changes are documented here.

## 0.3.1 — 2026-08-21

### OSS readiness

- Reframed the README around failure reconstruction, with verified installation
  and CLI commands above the fold.
- Added an offline failure/retry demo, self-contained HTML timeline, and browser-
  captured screenshot generated from the real trace.
- Bundled HTML templates, typing metadata, and neutral synthesis examples in the
  wheel; added clean-environment distribution smoke tests.
- Added Python 3.9–3.12 CI, contribution guidance, and public-path tests.
- Consolidated internal setup notes into project documentation.
- Added the installed `innertrace demo` command with deterministic offline trace
  and HTML output.
- Moved synthesis configuration to user-owned paths, retained a deprecated
  package-local compatibility fallback, and made no-config synthesis use the
  offline truncation provider.

## 0.3.0 — 2026-08-20

### Reconstruct the run

Version 0.3 presents the existing trace core as one coherent debugging path:
record execution facts locally, preserve their causal links, and project them
into a timeline or execution story.

- Added JSON-safe tool argument, result, and LLM tool-call capture.
- Added redaction before tool payload persistence and preview generation.
- Added incremental storytelling cursors for chat consumers.
- Preserved span kind on completion and cleared completed run context.
- Added the installable `innertrace` CLI and initial 0.3 package metadata.

The GitHub tag and release do not imply that a package was published to PyPI.

## 0.2.0

- Added opt-in function and block tracing.
- Added HTML index, run, and timeline templates.
- Added canonical and compact timeline projections.
- Added optional provider-based timeline synthesis.

## 0.1.0

- Introduced append-only run/span events and the content-addressed blob store.
- Added LLM, tool, router, sandbox, and exception events.
- Added the stable storytelling events and execution links.
