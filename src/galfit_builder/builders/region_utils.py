from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import math

if TYPE_CHECKING:
    from regions import EllipsePixelRegion
    from numpy.typing import NDArray


@dataclass(frozen=True)
class EllipseGeometry:
    """Extracted geometry from an ellipse region in GALFIT coordinates."""

    x_pos: float
    y_pos: float
    a: float  # semi-major axis
    b: float  # semi-minor axis
    pos_angle: float  # position angle in degrees


def ellipse_geometry(region: EllipsePixelRegion) -> EllipseGeometry:
    x = float(region.center.x) + 1
    y = float(region.center.y) + 1

    a = float(region.width) / 2.0
    b = float(region.height) / 2.0

    if a <= 0 or b <= 0:
        raise ValueError("Ellipse axes must be positive")

    angle = float(region.angle.to("deg").value)

    if b > a:
        a, b = b, a
        angle += 90

    return EllipseGeometry(
        x_pos=x,
        y_pos=y,
        a=a,
        b=b,
        pos_angle=(angle - 90) % 360,
    )


def get_local_patch(
    image_data: np.ndarray, x: int, y: int, size: int = 2
) -> np.ndarray:
    """Extract a local patch around (x, y) from image_data."""
    h, w = image_data.shape

    x0 = max(0, x - size)
    x1 = min(w, x + size + 1)

    y0 = max(0, y - size)
    y1 = min(h, y + size + 1)

    return image_data[y0:y1, x0:x1]


def estimate_ellipse_annulus_flux(
    region: EllipsePixelRegion,
    image_data: np.ndarray,
    radius: float,
    width: float = 1.0,
) -> tuple[float, bool]:

    geom = ellipse_geometry(region)

    x0 = geom.x_pos - 1
    y0 = geom.y_pos - 1

    q = geom.b / geom.a
    theta = np.deg2rad(geom.pos_angle)

    y_indices, x_indices = np.indices(image_data.shape)

    dx = x_indices - x0
    dy = y_indices - y0

    x_rot = dx * np.cos(theta) + dy * np.sin(theta)
    y_rot = dx * np.sin(theta) + dy * np.cos(theta)

    elliptical_radius = np.sqrt(x_rot**2 + (y_rot / q)**2)

    mask = (
        (elliptical_radius >= radius - width) &
        (elliptical_radius <= radius + width)
    )

    annulus_pixels = image_data[mask]
    valid_pixels = annulus_pixels[np.isfinite(annulus_pixels)]

    if valid_pixels.size == 0:
        return 0.0, True

    median_flux = float(np.median(valid_pixels))

    if median_flux <= 0:
        return 0.0, True

    return median_flux, False


def estimate_center_pixel_flux(
    region: EllipsePixelRegion,
    image_data: np.ndarray,
) -> tuple[float, bool]:

    geom = ellipse_geometry(region)
    x = int(round(geom.x_pos - 1))
    y = int(round(geom.y_pos - 1))

    inner_pixels = get_local_patch(image_data, x, y)
    valid_pixels = inner_pixels[np.isfinite(inner_pixels)]

    if valid_pixels.size == 0:
        return 0.0, True

    max_flux = float(np.max(valid_pixels))

    if max_flux <= 0:
        return 0.0, True

    return max_flux, False



HALF_LIGHT_TO_TOTAL_FLUX = 2.0


def estimate_ellipse_flux(
    region: EllipsePixelRegion,
    image_data: np.ndarray,
) -> tuple[float, bool]:

    mask = region.to_mask(mode="center")
    mask_image = mask.to_image(image_data.shape)

    if mask_image is None:
        return 0.0, True

    masked = mask_image * image_data
    flux = float(np.nansum(masked)) * HALF_LIGHT_TO_TOTAL_FLUX

    if flux <= 0:
        geom = ellipse_geometry(region)
        axis_ratio = geom.b / geom.a
        area = math.pi * geom.a * (geom.a * axis_ratio)

        x = int(round(geom.x_pos - 1))
        y = int(round(geom.y_pos - 1))

        if (0 <= y < image_data.shape[0] and 0 <= x < image_data.shape[1]):
            flux = float(area * image_data[y,x])
            return flux, True

        else:
            return 0.0, True

    return flux, False


def flux_to_mag(
        flux: float,
        zeropoint: float,
        fallback_mag: float
) -> float:

    if flux > 0:
        return -2.5 * math.log10(flux) + zeropoint

    return fallback_mag
