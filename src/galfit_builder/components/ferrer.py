from __future__ import annotations
from dataclasses import dataclass
from .base import GalfitComponent, GalfitParam


@dataclass
class Ferrer(GalfitComponent):
    component_number: int
    x_pos: GalfitParam
    y_pos: GalfitParam
    cen_surf_bright: GalfitParam
    outer_trunc_rad: GalfitParam
    alpha: GalfitParam
    beta: GalfitParam
    axis_ratio: GalfitParam
    pos_angle: GalfitParam
    include_in_output: bool = True


    def validate(self) -> None:
        PARAMS = [
            "x_pos",
            "y_pos",
            "cen_surf_bright",
            "outer_trunc_rad",
            "alpha",
            "beta",
            "axis_ratio",
            "pos_angle",
        ]

        for name in PARAMS:
            param = getattr(self, name)
            self._validate_param(name, param)

        if self.x_pos.value <= 0 or self.y_pos.value <= 0:
            raise ValueError("Ferrer position must be positive (GALFIT pixel coords)")

        if self.outer_trunc_rad.value <= 0:
            raise ValueError("Outer truncation radius must be positive")

        if self.alpha.value <= 0:
            raise ValueError("Alpha must be positive")

        if self.beta.value < 0:
            raise ValueError("Beta must be >= 0")

        if not (0 < self.axis_ratio.value <= 1):
            raise ValueError("Axis ratio must satisfy 0 < b/a <= 1")


    def to_galfit(self) -> str:
        self.validate()

        z = 0 if self.include_in_output else 1

        lines = []
        lines.append(f"\n# Object number: {self.component_number}")
        lines.append(" 0) ferrer                 # object type")

        lines.append(
            f" 1) {self.x_pos.value:.2f}  {self.y_pos.value:.2f}  "
            f"{self._fit_flag(self.x_pos)} {self._fit_flag(self.y_pos)}  "
            "# position x, y"
        )

        lines.append(
            f" 3) {self.cen_surf_bright.value:.4f}   "
            f"{self._fit_flag(self.cen_surf_bright)}  "
            "# central surface brightness [mag/arcsec^2]"
        )

        lines.append(
            f" 4) {self.outer_trunc_rad.value:.4f}   "
            f"{self._fit_flag(self.outer_trunc_rad)}  "
            "# outer truncation radius [pix]"
        )

        lines.append(
            f" 5) {self.alpha.value:.4f}   "
            f"{self._fit_flag(self.alpha)}  "
            "# alpha (outer truncation sharpness)"
        )

        lines.append(
            f" 6) {self.beta.value:.4f}   "
            f"{self._fit_flag(self.beta)}  "
            "# beta (central slope)"
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
            cen_surf_bright: float | int,
            outer_trunc_rad: float | int,
            alpha: float | int,
            beta: float | int,
            axis_ratio: float | int,
            pos_angle: float | int,
            include_in_output: bool = True
        ) -> Ferrer:
            return cls(
                component_number=component_number,
                x_pos=cls._ensure_param(x_pos),
                y_pos=cls._ensure_param(y_pos),
                cen_surf_bright=cls._ensure_param(cen_surf_bright),
                outer_trunc_rad=cls._ensure_param(outer_trunc_rad),
                alpha=cls._ensure_param(alpha),
                beta=cls._ensure_param(beta),
                axis_ratio=cls._ensure_param(axis_ratio),
                pos_angle=cls._ensure_param(pos_angle),
                include_in_output=include_in_output
            )

