from __future__ import annotations
from dataclasses import dataclass
from .base import GalfitComponent, GalfitParam


@dataclass
class Nuker(GalfitComponent):
    component_number: int
    x_pos: GalfitParam
    y_pos: GalfitParam
    mu: GalfitParam
    rb: GalfitParam
    alpha: GalfitParam
    beta: GalfitParam
    gamma: GalfitParam
    axis_ratio: GalfitParam
    pos_angle: GalfitParam
    include_in_output: bool = True


    def validate(self) -> None:
        PARAMS = ["x_pos", "y_pos", "mu", "rb", "alpha", "beta", "gamma", "axis_ratio", "pos_angle"]
        for name in PARAMS:
            param = getattr(self, name)
            self._validate_param(name, param)

        if self.x_pos.value <= 0 or self.y_pos.value <= 0:
            raise ValueError("Nuker position must be positive (GALFIT pixel coords)")

        if self.rb.value <= 0:
            raise ValueError("Rb must be > 0")

        if self.alpha.value <= 0:
            raise ValueError("Alpha must be > 0")

        if self.beta.value <= 0:
            raise ValueError("Beta must be > 0")

        if self.gamma.value < 0:
            raise ValueError("Gamma must be positive")

        if not (0 < self.axis_ratio.value <= 1):
            raise ValueError("Axis ratio must satisfy 0 < b/a <= 1")


    def to_galfit(self) -> str:
        self.validate()

        z = 0 if self.include_in_output else 1

        lines = []
        lines.append(f"\n# Object number: {self.component_number}")
        lines.append(" 0) nuker                  # object type")

        lines.append(
            f" 1) {self.x_pos.value:.2f}  {self.y_pos.value:.2f}  "
            f"{self._fit_flag(self.x_pos)} {self._fit_flag(self.y_pos)}  "
            "# position x, y"
        )

        lines.append(
            f" 3) {self.mu.value:.4f}   "
            f"{self._fit_flag(self.mu)}  "
            "# mu(Rb) [mag/arcsec^2]"
        )

        lines.append(
            f" 4) {self.rb.value:.4f}   "
            f"{self._fit_flag(self.rb)}  "
            "# Rb [pix]"
        )

        lines.append(
            f" 5) {self.alpha.value:.4f}   "
            f"{self._fit_flag(self.alpha)}  "
            "# alpha"
        )

        lines.append(
            f" 6) {self.beta.value:.4f}   "
            f"{self._fit_flag(self.beta)}  "
            "# beta"
        )

        lines.append(
            f" 7) {self.gamma.value:.4f}   "
            f"{self._fit_flag(self.gamma)}  "
            "# gamma"
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
