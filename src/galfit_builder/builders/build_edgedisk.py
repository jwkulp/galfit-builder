from __future__ import annotations
from typing import TYPE_CHECKING, Any
import warnings

from .builder_utils import ParamResolver
from ..components.base import GalfitParam
from ..components.edgedisk import Edgedisk
from .region_utils import ellipse_geometry, estimate_center_pixel_flux, flux_to_mag

if TYPE_CHECKING:
    import numpy as np
    from regions import EllipsePixelRegion


def build_edgedisk(
    component_number: int,
    region: EllipsePixelRegion,
    config: dict[str, Any],
    image_data: np.ndarray,
) -> Edgedisk:
    """Build an Edgedisk component from an ellipse region.

    For edge-on disks, the ellipse semi-major axis maps to the disk scale length,
    and the semi-minor axis maps to the disk scale height.
    """
    defaults = config.get("defaults", {}).get("edgedisk", {})
    overrides = config.get("overrides", {}).get("edgedisk", {})

    settings = ParamResolver(defaults, overrides)

    geom = ellipse_geometry(region)

    central_pixel_flux, used_fallback = estimate_center_pixel_flux(region, image_data)

    if used_fallback:
        warnings.warn("Central surface brightness estimation failed for edgedisk")

    zeropoint = config["galfit_input_controls"]["zeropoint"]
    fallback_mu = config["photometry"].get("fallback_mu", 25.0)

    plate_scale = config["galfit_input_controls"]["plate_scale"]
    plate_area = plate_scale[0] * plate_scale[1]

    surface_brightness_flux = central_pixel_flux / plate_area
    mu = flux_to_mag(surface_brightness_flux, zeropoint, fallback_mu)

    scale_length_ratio = settings.get("scale_length_ratio", 1.0)
    scale_height_ratio = settings.get("scale_height_ratio", 1.0)

    disk_scale_length = geom.a * scale_length_ratio
    disk_scale_height = geom.b * scale_height_ratio

    return Edgedisk(
        component_number=component_number,
        x_pos=GalfitParam(geom.x_pos, settings.get("x_pos_freeze", False)),
        y_pos=GalfitParam(geom.y_pos, settings.get("y_pos_freeze", False)),
        cen_surf_bright=GalfitParam(mu, settings.get("cen_surf_bright_freeze", False)),
        disk_scale_height=GalfitParam(disk_scale_height, settings.get("disk_scale_height_freeze", False)),
        disk_scale_length=GalfitParam(disk_scale_length, settings.get("disk_scale_length_freeze", False)),
        pos_angle=GalfitParam(geom.pos_angle, settings.get("pos_angle_freeze", False)),
        include_in_output=settings.get("include_in_output", True),
    )
