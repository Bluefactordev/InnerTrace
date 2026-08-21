import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_demo_module():
    path = ROOT / "examples" / "agent_failure_demo.py"
    spec = importlib.util.spec_from_file_location("agent_failure_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_readme_uses_installable_public_paths():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "from innertrace import Tracer" in readme
    assert "python examples/agent_failure_demo.py" in readme
    assert "innertrace --events-path" in readme
    assert "pip install innertrace" not in readme
    assert "from tracing" not in readme
    assert "./trace" not in readme


def test_offline_demo_records_failure_retry_and_final_result(tmp_path):
    demo = _load_demo_module()
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
