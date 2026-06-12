from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any
import math


@dataclass
class GalfitParam:
    """A single GALFIT parameter with value and freeze flag."""

    value: float
    freeze: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"Value must be numeric, got {type(self.value).__name__}")

        self.value = float(self.value)

        if not isinstance(self.freeze, bool):
            raise TypeError("Freeze must be a boolean")

        if not math.isfinite(self.value):
            raise ValueError(f"Value must be finite, got {self.value}")


class GalfitComponent(ABC):
    """Base class for all GALFIT component types."""

    component_number: int
    include_in_output: bool

    @staticmethod
    def _fit_flag(param: GalfitParam) -> int:
        """Return 0 if frozen, 1 if free to vary."""
        return 0 if param.freeze else 1

    def freeze_all_params(self) -> None:
        """Freeze all GalfitParam attributes."""
        for param in vars(self).values():
            if isinstance(param, GalfitParam):
                param.freeze = True

    def unfreeze_all_params(self) -> None:
        """Unfreeze all GalfitParam attributes."""
        for param in vars(self).values():
            if isinstance(param, GalfitParam):
                param.freeze = False

    @staticmethod
    def _validate_param(name: str, param: GalfitParam) -> None:
        """Validate that param is a GalfitParam instance."""
        if not isinstance(param, GalfitParam):
            raise TypeError(f"{name} must be a GalfitParam")

    @staticmethod
    def _ensure_param(value: float | int | GalfitParam) -> GalfitParam:
        """Convert a numeric value to GalfitParam if needed."""
        if isinstance(value, GalfitParam):
            return value
        return GalfitParam(float(value))

    @abstractmethod
    def to_galfit(self) -> str:
        """Generate GALFIT feedme component block."""
        pass

    def __str__(self) -> str:
        return self.to_galfit()
