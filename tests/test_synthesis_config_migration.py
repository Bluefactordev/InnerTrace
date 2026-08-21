import json
import os
import subprocess
import sys
from pathlib import Path

from innertrace.tracing import cli


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_user_config_precedence_and_explicit_missing_path(tmp_path, monkeypatch):
    cwd_config = tmp_path / ".innertrace" / "config.json"
    _write_json(cwd_config, {"source": "cwd"})
    explicit = tmp_path / "explicit.json"
    _write_json(explicit, {"source": "explicit"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INNERTRACE_CONFIG_PATH", str(explicit))

    assert cli.load_config() == {"source": "explicit"}

    monkeypatch.setenv("INNERTRACE_CONFIG_PATH", str(tmp_path / "missing.json"))
    assert cli.load_config() is None


def test_package_local_config_is_compatible_but_deprecated(
    tmp_path, monkeypatch, capsys
):
    package_dir = tmp_path / "package"
    package_config = package_dir / "config.json"
    _write_json(package_config, {"source": "legacy"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INNERTRACE_CONFIG_PATH", raising=False)
    monkeypatch.setattr(cli, "__file__", str(package_dir / "cli.py"))

    assert cli.load_config() == {"source": "legacy"}
    warning = capsys.readouterr().err
    assert "package-local synthesis configuration" in warning
    assert "deprecated" in warning


def test_no_config_synthesis_uses_offline_fallback(tmp_path):
    demo_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "innertrace.demo",
            "--output-dir",
            str(tmp_path / "demo"),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert demo_result.returncode == 0, demo_result.stderr

    timeline_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "innertrace.tracing.cli",
            "--events-path",
            str(tmp_path / "demo" / "traces" / "events.jsonl"),
            "timeline",
            "--last",
            "--synthesize",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert timeline_result.returncode == 0, timeline_result.stderr
    assert "offline truncation provider" in timeline_result.stderr
    assert "inventory.primary.lookup" in timeline_result.stdout


def test_distribution_inputs_contain_only_neutral_synthesis_examples():
    tracked_paths = [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ]
    package_config = ROOT / "innertrace" / "tracing" / "config.json"
    package_env = ROOT / "innertrace" / "tracing" / ".env"
    example = ROOT / "innertrace" / "tracing" / "config.json.example"

    assert not package_config.exists()
    assert not package_env.exists()
    assert json.loads(example.read_text(encoding="utf-8"))["default_quality"] == "low"

    forbidden = (
        "localhost:" + "5099",
        "qwen3" + "-30b",
        "gemma-3" + "-270",
        "gpt-oss" + "-120b",
    )
    searchable = [
        path
        for path in tracked_paths
        if path != Path(__file__)
        and path.suffix in {".py", ".json", ".md", ".toml", ".yml"}
    ]
    contents = "\n".join(path.read_text(encoding="utf-8") for path in searchable)
    assert all(value not in contents for value in forbidden)
