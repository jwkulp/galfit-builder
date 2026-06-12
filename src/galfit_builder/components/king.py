from dataclasses import dataclass
from .base import GalfitComponent, GalfitParam


@dataclass(kw_only=True)
class King(GalfitComponent):
    component_number: int
    x_pos: GalfitParam
    y_pos: GalfitParam
    mu: GalfitParam
    rc: GalfitParam
    rt: GalfitParam
    alpha: GalfitParam
    axis_ratio: GalfitParam
    pos_angle: GalfitParam
    include_in_output: bool = True


    def validate(self) -> None:
        PARAMS = [
            "x_pos",
            "y_pos",
            "mu",
            "rc",
            "rt",
            "alpha",
            "axis_ratio",
            "pos_angle",
        ]

        for name in PARAMS:
            param = getattr(self, name)
            self._validate_param(name, param)

        
        if self.x_pos.value <= 0 or self.y_pos.value <= 0:
            raise ValueError("King position must be positive (GALFIT pixel coords)")

        if self.rc.value <= 0:
            raise ValueError("King core radius rc must be positive")

        if self.rt.value <= 0:
            raise ValueError("King truncation radius rt must be positive")

        if self.rt.value <= self.rc.value:
            raise ValueError("King truncation radius rt must be larger than core radius rc")

        if self.alpha.value <= 0:
            raise ValueError("King alpha must be positive")

        if not (0 < self.axis_ratio.value <= 1):
            raise ValueError("Axis ratio must satisfy 0 < b/a <= 1")


    def to_galfit(self) -> str:
        self.validate()

        z = 0 if self.include_in_output else 1

        lines = []
        lines.append(f"\n# Object number: {self.component_number}")
        lines.append(" 0) king                   # object type")

        lines.append(
            f" 1) {self.x_pos.value:.2f}  {self.y_pos.value:.2f}  "
            f"{self._fit_flag(self.x_pos)} {self._fit_flag(self.y_pos)}  "
            "# position x, y"
        )

        lines.append(
            f" 3) {self.mu.value:.4f}   "
            f"{self._fit_flag(self.mu)}  "
            "# mu(0) [mag/arcsec^2]"
        )

        lines.append(
            f" 4) {self.rc.value:.4f}   "
            f"{self._fit_flag(self.rc)}  "
            "# Rc (core radius) [pix]"
        )

        lines.append(
            f" 5) {self.rt.value:.4f}   "
            f"{self._fit_flag(self.rt)}  "
            "# Rt (truncation radius) [pix]"
        )

        lines.append(
            f" 6) {self.alpha.value:.4f}   "
            f"{self._fit_flag(self.alpha)}  "
            "# alpha"
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
