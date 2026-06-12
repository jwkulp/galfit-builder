from __future__ import annotations
from typing import Any

AUTO = "auto"


class ParamResolver:
    """Resolve parameter values from defaults and overrides."""

    def __init__(self, defaults: dict[str, Any], overrides: dict[str, Any]) -> None:
        self.defaults = defaults
        self.overrides = overrides

    def get(self, key: str, fallback: Any = None) -> Any:
        """Get a parameter value, checking overrides first, then defaults."""
        value = self.overrides.get(key, AUTO)

        if value == AUTO:
            return self.defaults.get(key, fallback)

        return value
