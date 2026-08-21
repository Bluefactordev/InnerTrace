"""Access the HTML viewers bundled with the installed package."""

from importlib.resources import files
from typing import Tuple


_TEMPLATE_NAMES: Tuple[str, ...] = ("index.html", "runs.html", "timeline.html")


def template_names() -> Tuple[str, ...]:
    """Return the names of the bundled trace-view templates."""

    return _TEMPLATE_NAMES


def get_template_text(name: str, encoding: str = "utf-8") -> str:
    """Read a bundled template by public name."""

    if name not in _TEMPLATE_NAMES:
        raise ValueError(f"Unknown InnerTrace template: {name}")
    return files("innertrace").joinpath("templates", name).read_text(encoding=encoding)
