from __future__ import annotations
from dataclasses import dataclass
from .base import GalfitComponent, GalfitParam


@dataclass
class Edgedisk(GalfitComponent):
    component_number: int
    x_pos: GalfitParam
    y_pos: GalfitParam
    cen_surf_bright: GalfitParam
    disk_scale_height: GalfitParam
    disk_scale_length: GalfitParam
    pos_angle: GalfitParam
    include_in_output: bool = True


    def validate(self) -> None:
        PARAMS = ["x_pos",
                  "y_pos",
                  "cen_surf_bright",
                  "disk_scale_height",
                  "disk_scale_length",
                  "pos_angle"]

        for name in PARAMS:
            param = getattr(self, name)
            self._validate_param(name, param)

        if self.x_pos.value <= 0 or self.y_pos.value <= 0:
            raise ValueError("Edgedisk position must be positive (GALFIT pixel coords)")

        if self.disk_scale_height.value <= 0:
            raise ValueError("Disk scale height must be positive")

        if self.disk_scale_length.value <= 0:
            raise ValueError("Disk scale length must be positive")


    def to_galfit(self) -> str:
        self.validate()

        z = 0 if self.include_in_output else 1

        lines = []
        lines.append(f"\n# Object number: {self.component_number}")
        lines.append(" 0) edgedisk               # object type")

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
            f" 4) {self.disk_scale_height.value:.4f}   "
            f"{self._fit_flag(self.disk_scale_height)}  "
            "# disk scale-height [pix]"
        )

        lines.append(
            f" 5) {self.disk_scale_length.value:.4f}   "
            f"{self._fit_flag(self.disk_scale_length)}  "
            "# disk scale-length [pix]"
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
            disk_scale_height: float | int,
            disk_scale_length: float | int,
            pos_angle: float | int,
            include_in_output: bool = True
        ) -> Edgedisk:
            return cls(
                component_number=component_number,
                x_pos=cls._ensure_param(x_pos),
                y_pos=cls._ensure_param(y_pos),
                cen_surf_bright=cls._ensure_param(cen_surf_bright),
                disk_scale_height=cls._ensure_param(disk_scale_height),
                disk_scale_length=cls._ensure_param(disk_scale_length),
                pos_angle=cls._ensure_param(pos_angle),
                include_in_output=include_in_output
            )
