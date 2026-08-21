#!/usr/bin/env python3
"""Deterministic, offline failure-and-retry demo for InnerTrace."""

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from innertrace import Tracer, get_template_text
from innertrace.tracing import (
    emit_router_decision,
    emit_tool_call_end,
    emit_tool_call_start,
)
from innertrace.tracing.projections import list_runs, timeline_view


class StaleInventoryError(RuntimeError):
    """Raised when the simulated cache returns a known-stale answer."""


@dataclass(frozen=True)
class DemoResult:
    run_id: str
    events_path: Path
    html_path: Path
    final_result: Dict[str, object]


def run_demo(output_dir: Path, html_path: Optional[Path] = None) -> DemoResult:
    """Run the fixed agent scenario and build its self-contained HTML viewer."""

    output_dir = output_dir.resolve()
    events_path = output_dir / "traces" / "events.jsonl"
    blobs_path = output_dir / "traces" / "blobs"
    html_path = (html_path or output_dir / "agent-failure.html").resolve()

    events_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    if events_path.exists():
        events_path.unlink()

    tracer = Tracer(events_path=str(events_path), blobs_path=str(blobs_path))
    run_id = tracer.start_run(
        entrypoint="demo.inventory_agent",
        args={"request": "How many CHAIR-42 units are available?"},
    )

    final_result: Dict[str, object]
    try:
        with tracer.span("answer_inventory_question", actor="agent.inventory", kind="agent"):
            with tracer.span("plan_lookup", actor="agent.planner", kind="agent"):
                emit_router_decision(
                    tracer,
                    rule="prefer_low_latency_source",
                    candidates=["inventory.cache", "inventory.primary"],
                    chosen="inventory.cache",
                    why="The cache is normally the fastest inventory source.",
                    actor="router.inventory",
                )

            try:
                with tracer.span(
                    "cached_inventory_branch",
                    actor="tool.inventory.cache",
                    kind="tool",
                    tags=["attempt:1", "branch:cache"],
                ):
                    emit_tool_call_start(
                        tracer,
                        "inventory.cache.lookup",
                        {"sku": "CHAIR-42"},
                        actor="tool.inventory.cache",
                    )
                    stale_result = {"sku": "CHAIR-42", "available": 0, "age_seconds": 901}
                    emit_tool_call_end(
                        tracer,
                        "inventory.cache.lookup",
                        stale_result,
                        status="ok",
                        latency_ms=2,
                        actor="tool.inventory.cache",
                    )
                    raise StaleInventoryError(
                        "cache age 901s exceeds the 300s freshness limit"
                    )
            except StaleInventoryError as error:
                tracer.emit(
                    type="agent.retry",
                    actor="agent.inventory",
                    level="warn",
                    tags=["retry", "attempt:2"],
                    payload={
                        "attempt": 2,
                        "reason": str(error),
                        "from": "inventory.cache",
                        "to": "inventory.primary",
                    },
                )

            with tracer.span(
                "primary_inventory_branch",
                actor="tool.inventory.primary",
                kind="tool",
                tags=["attempt:2", "branch:primary"],
            ):
                emit_router_decision(
                    tracer,
                    rule="retry_after_stale_cache",
                    candidates=["inventory.primary"],
                    chosen="inventory.primary",
                    actor="router.inventory",
                )
                emit_tool_call_start(
                    tracer,
                    "inventory.primary.lookup",
                    {"sku": "CHAIR-42"},
                    actor="tool.inventory.primary",
                )
                final_result = {
                    "sku": "CHAIR-42",
                    "available": 7,
                    "source": "inventory.primary",
                }
                emit_tool_call_end(
                    tracer,
                    "inventory.primary.lookup",
                    final_result,
                    status="ok",
                    latency_ms=4,
                    actor="tool.inventory.primary",
                )

            tracer.emit(
                type="agent.result",
                actor="agent.inventory",
                payload=final_result,
            )
    except Exception:
        tracer.end_run(status="error")
        raise
    else:
        tracer.end_run(status="ok")

    _write_html_demo(run_id, events_path, html_path)
    return DemoResult(run_id, events_path, html_path, final_result)


def _write_html_demo(run_id: str, events_path: Path, html_path: Path) -> None:
    """Render the existing timeline template from real projection output."""

    runs = list_runs(str(events_path), limit=1)
    if not runs or runs[0]["run_id"] != run_id:
        raise RuntimeError("The generated run could not be projected")

    projection = {
        "run_id": run_id,
        "compact": False,
        "synthesize": False,
        "timeline": timeline_view(run_id, str(events_path)),
    }
    embedded = {"run": runs[0], "timeline": projection}
    serialized = json.dumps(embedded, ensure_ascii=False).replace("</", "<\\/")
    script = (
        '<script id="innertrace-demo-data" type="application/json">'
        f"{serialized}</script>"
    )
    template = get_template_text("timeline.html")
    marker = "<!-- INNERTRACE_EMBEDDED_DEMO -->"
    if marker not in template:
        raise RuntimeError("The bundled timeline template has no demo marker")
    html_path.write_text(template.replace(marker, script), encoding="utf-8")


def capture_screenshot(html_path: Path, screenshot_path: Path) -> None:
    """Capture the generated page with a locally installed Chromium browser."""

    browser = next(
        (
            path
            for executable in ("google-chrome", "chromium", "chromium-browser")
            if (path := shutil.which(executable))
        ),
        None,
    )
    if browser is None:
        raise RuntimeError(
            "No Chromium browser found. The HTML demo was generated; install "
            "Chrome or Chromium to refresh the PNG."
        )

    screenshot_path = screenshot_path.resolve()
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            browser,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--hide-scrollbars",
            "--run-all-compositor-stages-before-draw",
            "--window-size=1500,1000",
            f"--screenshot={screenshot_path}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_output"),
        help="Trace output directory (default: demo_output)",
    )
    parser.add_argument(
        "--html",
        type=Path,
        help="HTML destination (default: <output-dir>/agent-failure.html)",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Optional PNG destination captured from the generated HTML",
    )
    args = parser.parse_args()

    result = run_demo(args.output_dir, args.html)
    if args.screenshot:
        capture_screenshot(result.html_path, args.screenshot)

    print(f"Run ID: {result.run_id}")
    print(f"Trace: {result.events_path}")
    print(f"HTML demo: {result.html_path}")
    if args.screenshot:
        print(f"Screenshot: {args.screenshot.resolve()}")
    print("Final result: CHAIR-42 has 7 units (source: inventory.primary)")
    print("Inspect the causal timeline with:")
    print(
        "  innertrace --events-path "
        f"{result.events_path} timeline --last"
    )


if __name__ == "__main__":
    main()
