from __future__ import annotations
from dataclasses import dataclass
from .base import GalfitComponent, GalfitParam


@dataclass
class Gaussian(GalfitComponent):
    component_number: int
    x_pos: GalfitParam
    y_pos: GalfitParam
    mag: GalfitParam
    fwhm: GalfitParam
    axis_ratio: GalfitParam
    pos_angle: GalfitParam
    include_in_output: bool = True


    def validate(self) -> None:
        PARAMS = [
            "x_pos",
            "y_pos",
            "mag",
            "fwhm",
            "axis_ratio",
            "pos_angle",
        ]

        for name in PARAMS:
            param = getattr(self, name)
            self._validate_param(name, param)
        
        if self.x_pos.value <= 0 or self.y_pos.value <= 0:
            raise ValueError("Gaussian position msut be positive (GALFIT pixel coords)")

        if self.fwhm.value <= 0:
            raise ValueError("FWHM must be positive")

        if not (0 < self.axis_ratio.value <= 1):
            raise ValueError("Axis ratio must satisfy 0 < b/a <= 1")


    def to_galfit(self) -> str:
        self.validate()

        z = 0 if self.include_in_output else 1

        lines = []
        lines.append(f"\n# Object number: {self.component_number}")
        lines.append(" 0) gaussian               # object type")

        lines.append(
            f" 1) {self.x_pos.value:.2f}  {self.y_pos.value:.2f}  "
            f"{self._fit_flag(self.x_pos)} {self._fit_flag(self.y_pos)}  "
            "# position x, y"
        )

        lines.append(
            f" 3) {self.mag.value:.4f}   "
            f"{self._fit_flag(self.mag)}  "
            "# total magnitude"
        )

        lines.append(
            f" 4) {self.fwhm.value:.4f}   "
            f"{self._fit_flag(self.fwhm)}  "
            "# FWHM [pix]"
        )

        lines.append(
            f" 9) {self.axis_ratio.value:.4f}   "
            f"{self._fit_flag(self.axis_ratio)}  "
            "# axis ratio (b/a)"
        )

        lines.append(
            f"10) {self.pos_angle.value:.4f}   "
            f"{self._fit_flag(self.pos_angle)}  "
            "# position angle (PA) [deg: Up=0, Left=90]"
        )

        lines.append(f" Z) {z}                    # output option (0 = resid., 1 = Don't subtract)")

        return "\n".join(lines)


    @classmethod
    def from_values(
            cls,
            *,
            component_number: int,
            x_pos: float | int,
            y_pos: float | int,
            mag: float | int,
            fwhm: float | int,
            axis_ratio: float | int,
            pos_angle: float | int,
            include_in_output: bool = True
        ) -> Gaussian:
            return cls(
                component_number=component_number,
                x_pos=cls._ensure_param(x_pos),
                y_pos=cls._ensure_param(y_pos),
                mag=cls._ensure_param(mag),
                fwhm=cls._ensure_param(fwhm),
                axis_ratio=cls._ensure_param(axis_ratio),
                pos_angle=cls._ensure_param(pos_angle),
                include_in_output=include_in_output
            )
