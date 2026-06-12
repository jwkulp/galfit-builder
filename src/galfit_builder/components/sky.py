from __future__ import annotations
from dataclasses import dataclass
from .base import GalfitComponent, GalfitParam


@dataclass
class Sky(GalfitComponent):
    component_number: int
    background: GalfitParam
    dsky_dx: GalfitParam
    dsky_dy: GalfitParam
    include_in_output: bool = True

    def validate(self) -> None:
        self._validate_param("background", self.background)
        self._validate_param("dsky_dx", self.dsky_dx)
        self._validate_param("dsky_dy", self.dsky_dy)


    def to_galfit(self) -> str:
        self.validate()

        z = 0 if self.include_in_output else 1

        lines = []
        lines.append(f"\n# Object number: {self.component_number}")
        lines.append(" 0) sky                    # object type")
        lines.append(
            f" 1) {self.background.value:.4f}   "
            f"{self._fit_flag(self.background)}  "
            "# sky background at center of fitting region [ADUs]")

        lines.append(
            f" 2) {self.dsky_dx.value:.4f}   "
            f"{self._fit_flag(self.dsky_dx)}  "
            "# dsky/dx (sky gradient in x)"
        )

        lines.append(
            f" 3) {self.dsky_dy.value:.4f}   "
            f"{self._fit_flag(self.dsky_dy)}  "
            "# dsky/dy (sky gradient in y)"
        )

        lines.append(f" Z) {z}                    # output option (0 = resid., 1 = Don't subtract)")

        return "\n".join(lines)

    
    @classmethod
    def from_values(
        cls,
        *,
        component_number: int,
        background: float | int,
        dsky_dx: float | int,
        dsky_dy: float | int,
        include_in_output: bool = True
    ) -> Sky:
        return cls(
            component_number=component_number,
            background=cls._ensure_param(background),
            dsky_dx=cls._ensure_param(dsky_dx),
            dsky_dy=cls._ensure_param(dsky_dy),
            include_in_output=include_in_output
        )

