from __future__ import annotations
from dataclasses import dataclass
from .base import GalfitComponent, GalfitParam


@dataclass(kw_only=True)
class PSF(GalfitComponent):
    component_number: int
    x_pos: GalfitParam
    y_pos: GalfitParam
    mag: GalfitParam
    include_in_output: bool = True


    def validate(self) -> None:
        PARAMS = ["x_pos", "y_pos", "mag"]
        for name in PARAMS:
            param = getattr(self, name)
            self._validate_param(name, param)

        if self.x_pos.value <= 0 or self.y_pos.value <= 0:
            raise ValueError("PSF position must be positive (GALFIT pixel coords)")


    def to_galfit(self) -> str:
        self.validate()

        z = 0 if self.include_in_output else 1

        lines = []
        lines.append(f"\n# Object number: {self.component_number}")
        lines.append(" 0) psf                    # object type")

        lines.append(
            f" 1) {self.x_pos.value:.2f}  {self.y_pos.value:.2f}  "
            f"{self._fit_flag(self.x_pos)} {self._fit_flag(self.y_pos)}  "
            "# position x, y"
        )

        lines.append(
            f" 3) {self.mag.value:.3f}   "
            f"{self._fit_flag(self.mag)}  "
            "# total magnitude"
        )

        lines.append(f" Z) {z}                    # output option (0 = resid., 1 = Don't subtract)")

        return "\n".join(lines)
