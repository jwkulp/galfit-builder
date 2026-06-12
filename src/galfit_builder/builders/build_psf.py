from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .builder_utils import ParamResolver
from ..components.base import GalfitParam
from ..components.psf import PSF

if TYPE_CHECKING:
    import numpy as np
    from regions import PointPixelRegion


def build_psf(
    component_number: int,
    region: PointPixelRegion,
    config: dict[str, Any],
    image_data: np.ndarray | None = None,
) -> PSF:
    defaults = config["defaults"]["psf"]
    overrides = config.get("overrides", {}).get("psf", {})

    settings = ParamResolver(defaults, overrides)

    x = region.center.x + 1
    y = region.center.y + 1

    return PSF(
        component_number=component_number,
        x_pos=GalfitParam(x, settings.get("x_pos_freeze")),
        y_pos=GalfitParam(y, settings.get("y_pos_freeze")),
        mag=GalfitParam(settings.get("mag"), settings.get("mag_freeze")),
        include_in_output=settings.get("include_in_output")
    )
