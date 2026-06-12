from __future__ import annotations
import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path = "config.toml") -> dict[str, Any]:
    """Load and return configuration from a TOML file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        return tomllib.load(f)
