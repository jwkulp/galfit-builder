from __future__ import annotations
from typing import TYPE_CHECKING, Any
import warnings

from .builder_utils import ParamResolver
from ..components.base import GalfitParam
from ..components.gaussian import Gaussian
from .region_utils import ellipse_geometry, estimate_ellipse_flux, flux_to_mag

if TYPE_CHECKING:
    import numpy as np
    from regions import EllipsePixelRegion

FWHM_TO_SIGMA = 2.355


def build_gaussian(
    component_number: int,
    region: EllipsePixelRegion,
    config: dict[str, Any],
    image_data: np.ndarray,
) -> Gaussian:
    defaults = config.get("defaults", {}).get("gaussian", {})
    overrides = config.get("overrides", {}).get("gaussian", {})

    settings = ParamResolver(defaults, overrides)

    geom = ellipse_geometry(region)
    flux, used_fallback = estimate_ellipse_flux(region, image_data)

    if used_fallback and flux == 0.0:
        warnings.warn("Flux estimation failed completely")
    elif used_fallback:
        warnings.warn("Used central pixel estimation")

    zeropoint = config["galfit_input_controls"]["zeropoint"]
    fallback_mag = config["photometry"]["fallback_mag"]

    mag = flux_to_mag(flux, zeropoint, fallback_mag)
    axis_ratio = geom.b / geom.a

    fwhm = geom.a * FWHM_TO_SIGMA

    return Gaussian(
        component_number=component_number,
        x_pos=GalfitParam(geom.x_pos, settings.get("x_pos_freeze")),
        y_pos=GalfitParam(geom.y_pos, settings.get("y_pos_freeze")),
        mag=GalfitParam(mag, settings.get("mag_freeze")),
        fwhm=GalfitParam(fwhm, settings.get("fwhm_freeze")),
        axis_ratio=GalfitParam(axis_ratio, settings.get("axis_ratio_freeze")),
        pos_angle=GalfitParam(geom.pos_angle, settings.get("pos_angle_freeze")),
        include_in_output=settings.get("include_in_output"),
    )
