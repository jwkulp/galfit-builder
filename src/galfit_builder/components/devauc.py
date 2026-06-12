from .base import GalfitComponent, GalfitParam
from dataclasses import dataclass


@dataclass(kw_only=True)
class Devauc(GalfitComponent):
    component_number: int
    x_pos: GalfitParam
    y_pos: GalfitParam
    mag: GalfitParam
    eff_rad: GalfitParam
    axis_ratio: GalfitParam
    pos_angle: GalfitParam
    include_in_output: bool = True

    def validate(self) -> None:
        PARAMS = ["x_pos", "y_pos", "mag", "eff_rad", "axis_ratio", "pos_angle"]
        for name in PARAMS:
            self._validate_param(name, getattr(self, name))

        if self.x_pos.value <= 0 or self.y_pos.value <= 0:
            raise ValueError("Devauc position must be positive")

        if self.eff_rad.value <= 0:
            raise ValueError("Effective radius must be positive")

        if not (0 < self.axis_ratio.value <= 1):
            raise ValueError("Axis ratio must satisfy 0 < b/a <= 1")

    def to_galfit(self) -> str:
        self.validate()
        z = 0 if self.include_in_output else 1

        lines = []
        lines.append(f"\n# Object number: {self.component_number}")
        lines.append(" 0) devauc                 # object type")
        lines.append(
            f" 1) {self.x_pos.value:.2f}  {self.y_pos.value:.2f}  "
            f"{self._fit_flag(self.x_pos)} {self._fit_flag(self.y_pos)}  "
            "# position x, y"
        )
        lines.append(f" 3) {self.mag.value:.4f}   {self._fit_flag(self.mag)}  # integrated magnitude")
        lines.append(f" 4) {self.eff_rad.value:.4f}   {self._fit_flag(self.eff_rad)}  # R_e (effective radius) [pix]")
        lines.append(f" 9) {self.axis_ratio.value:.4f}   {self._fit_flag(self.axis_ratio)}  # axis ratio (b/a)")
        lines.append(f"10) {self.pos_angle.value:.4f}   {self._fit_flag(self.pos_angle)}  # position angle (PA) [deg: Up=0, Left=90]")
        lines.append(f" Z) {z}                    # output option (0 = resid., 1 = Don't subtract)")

        return "\n".join(lines)
