from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable

from regions import PointPixelRegion, EllipsePixelRegion, PixelRegion

from .build_psf import build_psf
from .build_sersic import build_sersic
from .build_devauc import build_devauc
from .build_edgedisk import build_edgedisk
from .build_expdisk import build_expdisk
from .build_ferrer import build_ferrer
from .build_gaussian import build_gaussian
from .build_king import build_king
from .build_moffat import build_moffat
from .build_nuker import build_nuker
from .build_sky import build_sky

if TYPE_CHECKING:
    import numpy as np
    from galfit_builder.components.base import GalfitComponent

BuilderFunc = Callable[[int, PixelRegion, dict[str, Any], "np.ndarray"], "GalfitComponent"]

BUILDERS: dict[str, BuilderFunc] = {
    "psf": build_psf,
    "sersic": build_sersic,
    "devauc": build_devauc,
    "edgedisk": build_edgedisk,
    "expdisk": build_expdisk,
    "ferrer": build_ferrer,
    "gaussian": build_gaussian,
    "king": build_king,
    "moffat": build_moffat,
    "nuker": build_nuker,
}


def infer_component_type(region: PixelRegion, color_map: dict[str, str]) -> str | None:
    """Determine GALFIT component type from region shape and color."""
    if isinstance(region, PointPixelRegion):
        return "psf"

    if isinstance(region, EllipsePixelRegion):
        # regions library uses edgecolor/facecolor for shapes, color for points
        color = region.visual.get("edgecolor") or region.visual.get("facecolor") or region.visual.get("color")

        if color is None:
            return "sersic"

        color = color.lower()

        if color not in color_map:
            raise ValueError(f"Unknown region color: {color}")

        return color_map[color]

    return None


def build_components(
    regions: list[PixelRegion],
    image_data: np.ndarray,
    config: dict[str, Any],
) -> list[GalfitComponent]:
    """Build GALFIT components from a list of DS9 regions.

    Sky is added first (component 1) if enabled in config, then region-based components follow.

    Args:
        regions: List of DS9 pixel regions
        image_data: FITS image data array
        config: Configuration dictionary

    Returns:
        List of GalfitComponent objects
    """
    components: list[GalfitComponent] = []
    color_map = config["region_colors"]

    component_number = 1

    sky_defaults = config.get("defaults", {}).get("sky", {})
    include_sky = sky_defaults.get("include", True)

    if include_sky:
        sky = build_sky(component_number, None, config, image_data)
        components.append(sky)
        component_number += 1

    for region in regions:
        component_type = infer_component_type(region, color_map)

        if component_type is None:
            continue

        if component_type not in BUILDERS:
            raise ValueError(f"Unsupported component type: {component_type}")

        builder = BUILDERS[component_type]
        comp = builder(component_number, region, config, image_data)
        components.append(comp)
        component_number += 1

    return components
