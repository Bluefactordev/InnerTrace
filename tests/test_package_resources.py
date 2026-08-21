from importlib.metadata import version
from pathlib import Path

import pytest

from innertrace import __version__, get_template_text, template_names


ROOT = Path(__file__).resolve().parents[1]


def test_version_and_templates_are_public_resources():
    assert __version__ == "0.3.0"
    assert version("innertrace") == __version__
    assert template_names() == ("index.html", "runs.html", "timeline.html")

    for name in template_names():
        packaged = get_template_text(name)
        source_compatibility_copy = (ROOT / "templates" / name).read_text(
            encoding="utf-8"
        )
        assert packaged.splitlines() == [
            line.rstrip() for line in source_compatibility_copy.splitlines()
        ]
        assert "<!DOCTYPE html>" in packaged


def test_template_reader_rejects_unknown_paths():
    with pytest.raises(ValueError, match="Unknown InnerTrace template"):
        get_template_text("../README.md")
