from __future__ import annotations
from typing import TYPE_CHECKING, Any
import warnings

from .builder_utils import ParamResolver
from ..components.base import GalfitParam
from ..components.king import King
from .region_utils import ellipse_geometry, estimate_center_pixel_flux, flux_to_mag

if TYPE_CHECKING:
    import numpy as np
    from regions import EllipsePixelRegion


def build_king(
    component_number: int,
    region: EllipsePixelRegion,
    config: dict[str, Any],
    image_data: np.ndarray,
) -> King:
    defaults = config["defaults"]["king"]
    overrides = config.get("overrides", {}).get("king", {})

    settings = ParamResolver(defaults, overrides)

    geom = ellipse_geometry(region)
    central_pixel_flux, used_fallback = estimate_center_pixel_flux(region, image_data)

    if used_fallback:
        warnings.warn("Central surface brightness estimation failed completely")

    zeropoint = config["galfit_input_controls"]["zeropoint"]
    fallback_mu = config["photometry"]["fallback_mu"]

    plate_scale = config["galfit_input_controls"]["plate_scale"]
    plate_area = plate_scale[0] * plate_scale[1]

    surface_brightness_flux = central_pixel_flux / plate_area

    mu = flux_to_mag(surface_brightness_flux, zeropoint, fallback_mu)

    rc = max(
        geom.a * settings.get("rc_ratio"),
        settings.get("min_rc")
    )

    rt = max(
        geom.a * settings.get("rt_ratio"),
        settings.get("min_rt")
    )

    axis_ratio = geom.b / geom.a

    return King(
        component_number=component_number,
        x_pos=GalfitParam(geom.x_pos, settings.get("x_pos_freeze")),
        y_pos=GalfitParam(geom.y_pos, settings.get("y_pos_freeze")), 
        mu=GalfitParam(mu, settings.get("mu_freeze")),
        rc=GalfitParam(rc, settings.get("rc_freeze")),
        rt=GalfitParam(rt, settings.get("rt_freeze")),
        alpha=GalfitParam(settings.get("alpha"), settings.get("alpha_freeze")),
        axis_ratio=GalfitParam(axis_ratio, settings.get("axis_ratio_freeze")),
        pos_angle=GalfitParam(geom.pos_angle, settings.get("pos_angle_freeze")),
        include_in_output=settings.get("include_in_output")
    )
