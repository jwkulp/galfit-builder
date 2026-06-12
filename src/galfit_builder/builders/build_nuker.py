from __future__ import annotations
from typing import TYPE_CHECKING, Any
import warnings

from .builder_utils import ParamResolver
from ..components.base import GalfitParam
from ..components.nuker import Nuker
from .region_utils import ellipse_geometry, estimate_ellipse_annulus_flux, estimate_center_pixel_flux, flux_to_mag

if TYPE_CHECKING:
    import numpy as np
    from regions import EllipsePixelRegion


def build_nuker(
    component_number: int,
    region: EllipsePixelRegion,
    config: dict[str, Any],
    image_data: np.ndarray,
) -> Nuker:
    defaults = config["defaults"]["nuker"]
    overrides = config.get("overrides", {}).get("nuker", {})

    settings = ParamResolver(defaults, overrides)

    geom = ellipse_geometry(region)

    rb = max(
        geom.a * settings.get("rb_ratio"),
        settings.get("min_rb")
    )

    rb_flux, used_fallback = estimate_ellipse_annulus_flux(region, image_data, rb)

    if used_fallback:
        warnings.warn("Nuker rb surface brightness estimate failed; using center flux")
        rb_flux, used_fallback = estimate_center_pixel_flux(region, image_data)

    if used_fallback:
        warnings.warn("Nuker surface brightness estimation failed completely")

    zeropoint = config["galfit_input_controls"]["zeropoint"]
    fallback_mu = config["photometry"]["fallback_mu"]

    plate_scale = config["galfit_input_controls"]["plate_scale"]
    plate_area = plate_scale[0] * plate_scale[1]

    surface_brightness_flux = rb_flux / plate_area
    mu = flux_to_mag(surface_brightness_flux, zeropoint, fallback_mu)

    axis_ratio = geom.b / geom.a

    return Nuker(
        component_number=component_number,
        x_pos=GalfitParam(geom.x_pos, settings.get("x_pos_freeze")),
        y_pos=GalfitParam(geom.y_pos, settings.get("y_pos_freeze")),
        mu=GalfitParam(mu, settings.get("mu_freeze")),
        rb=GalfitParam(rb, settings.get("rb_freeze")),
        alpha=GalfitParam(settings.get("alpha"), settings.get("alpha_freeze")),
        beta=GalfitParam(settings.get("beta"), settings.get("beta_freeze")),
        gamma=GalfitParam(settings.get("gamma"), settings.get("gamma_freeze")),
        axis_ratio=GalfitParam(axis_ratio, settings.get("axis_ratio_freeze")),
        pos_angle=GalfitParam(geom.pos_angle, settings.get("pos_angle_freeze")),
        include_in_output=settings.get("include_in_output"),
    )
