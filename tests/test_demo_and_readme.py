import json
import os
import subprocess
import sys
from pathlib import Path

from innertrace import demo


ROOT = Path(__file__).resolve().parents[1]


def _semantic_events(path: Path):
    return [
        {
            "type": event["type"],
            "actor": event.get("actor"),
            "level": event.get("level"),
            "tags": event.get("tags"),
            "payload": {
                key: value
                for key, value in event.get("payload", {}).items()
                if key != "args_ref"
                and not key.endswith("_ref")
                and not (event["type"] == "span.end" and key == "latency_ms")
            },
        }
        for event in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    ]


def test_readme_uses_installable_public_paths():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "from innertrace import Tracer" in readme
    assert 'python -m pip install "innertrace @ git+https://github.com/' in readme
    assert "innertrace demo" in readme
    assert "innertrace --events-path" in readme
    assert "from tracing" not in readme
    assert "./trace" not in readme
    assert "small fit" not in readme


def test_offline_demo_records_failure_retry_and_final_result(tmp_path):
    result = demo.run_demo(tmp_path)

    events = [
        json.loads(line)
        for line in result.events_path.read_text(encoding="utf-8").splitlines()
    ]
    event_types = [event["type"] for event in events]
    router_choices = [
        event["payload"]["chosen"]
        for event in events
        if event["type"] == "router.decision"
    ]

    assert event_types[0] == "run.start"
    assert "tool.call.start" in event_types
    assert "exception" in event_types
    assert "agent.retry" in event_types
    assert event_types[-1] == "run.end"
    assert router_choices == ["inventory.cache", "inventory.primary"]
    assert events[-1]["payload"]["status"] == "ok"
    assert result.final_result == {
        "sku": "CHAIR-42",
        "available": 7,
        "source": "inventory.primary",
    }

    html = result.html_path.read_text(encoding="utf-8")
    assert 'id="innertrace-demo-data"' in html
    assert "StaleInventoryError" in html
    assert "inventory.primary.lookup" in html


def test_demo_semantics_are_deterministic(tmp_path):
    first = demo.run_demo(tmp_path / "first")
    second = demo.run_demo(tmp_path / "second")

    assert _semantic_events(first.events_path) == _semantic_events(second.events_path)
    assert first.final_result == second.final_result


def test_installed_cli_demo_path(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    output_dir = tmp_path / "cli-demo"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "innertrace.tracing.cli",
            "demo",
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Final result: CHAIR-42 has 7 units" in result.stdout
    assert (output_dir / "traces" / "events.jsonl").is_file()
    assert (output_dir / "agent-failure.html").is_file()


def test_demo_screenshot_error_is_clear(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(demo.shutil, "which", lambda executable: None)

    status = demo.main(
        [
            "--output-dir",
            str(tmp_path),
            "--screenshot",
            str(tmp_path / "demo.png"),
        ]
    )

    assert status == 1
    assert "InnerTrace demo failed: No Chromium browser found" in capsys.readouterr().err
