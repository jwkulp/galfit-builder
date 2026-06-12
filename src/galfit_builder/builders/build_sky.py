from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .builder_utils import ParamResolver
from ..components.base import GalfitParam
from ..components.sky import Sky

if TYPE_CHECKING:
    import numpy as np
    from regions import PixelRegion


def estimate_sky_background(image_data: np.ndarray, edge_fraction: float = 0.1) -> float:
    """Estimate sky background from image edges using sigma-clipped median."""
    import numpy as np

    h, w = image_data.shape
    edge_h = max(1, int(h * edge_fraction))
    edge_w = max(1, int(w * edge_fraction))

    top = image_data[:edge_h, :].flatten()
    bottom = image_data[-edge_h:, :].flatten()
    left = image_data[edge_h:-edge_h, :edge_w].flatten()
    right = image_data[edge_h:-edge_h, -edge_w:].flatten()

    edge_pixels = np.concatenate([top, bottom, left, right])

    median = np.nanmedian(edge_pixels)
    if not np.isfinite(median):
        return 0.0

    std = np.nanstd(edge_pixels)
    clipped = edge_pixels[np.abs(edge_pixels - median) < 3 * std]

    if clipped.size == 0:
        return float(median)

    return float(np.nanmedian(clipped))


def build_sky(
    component_number: int,
    region: PixelRegion | None,
    config: dict[str, Any],
    image_data: np.ndarray,
) -> Sky:
    """Build a Sky component.

    If region is None, estimates sky from image edges.
    If region is provided, uses the region's visual properties or config values.
    """
    defaults = config.get("defaults", {}).get("sky", {})
    overrides = config.get("overrides", {}).get("sky", {})

    settings = ParamResolver(defaults, overrides)

    background = settings.get("background")
    if background is None or background == "auto":
        edge_fraction = settings.get("edge_fraction", 0.1)
        background = estimate_sky_background(image_data, edge_fraction)

    dsky_dx = settings.get("dsky_dx", 0.0)
    dsky_dy = settings.get("dsky_dy", 0.0)

    return Sky(
        component_number=component_number,
        background=GalfitParam(background, settings.get("background_freeze", False)),
        dsky_dx=GalfitParam(dsky_dx, settings.get("dsky_dx_freeze", True)),
        dsky_dy=GalfitParam(dsky_dy, settings.get("dsky_dy_freeze", True)),
        include_in_output=settings.get("include_in_output", True),
    )
