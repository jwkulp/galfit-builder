"""Cut and rotate FITS images based on a DS9 box region.

Use case: Smaller cutouts run faster in GALFIT. This tool:
1. Reads a box region from a DS9 region file
2. Rotates images to align with the box (if box is rotated)
3. Cuts out the box region
4. Rotates PSF to match
5. Handles sigma images (optional inverse variance conversion)

Headers (including WCS) are preserved and updated for the cutout.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from regions import Regions
from scipy.ndimage import rotate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cut and rotate FITS images based on a DS9 box region"
    )
    parser.add_argument("region", help="DS9 region file with box defining cutout area")
    parser.add_argument("--science", "-s", nargs="*", default=[],
                        help="Science FITS files to cut")
    parser.add_argument("--sigma", "-e", nargs="*", default=[],
                        help="Sigma/error FITS files to cut")
    parser.add_argument("--psf", "-p", nargs="*", default=[],
                        help="PSF FITS files to rotate (not cut)")
    parser.add_argument("--reference", "-r",
                        help="Reference file for image center (required if only rotating PSF)")
    parser.add_argument("--inverse-variance", "-v", action="store_true",
                        help="Sigma files are inverse variance; convert to sigma")
    parser.add_argument("--prefix", default="cutout_",
                        help="Output file prefix (default: cutout_)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite existing files")
    return parser.parse_args()


def get_image_center(fits_path: Path) -> tuple[float, float]:
    """Get image center from FITS header."""
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        xc = header["NAXIS1"] / 2
        yc = header["NAXIS2"] / 2
        return xc, yc


def parse_box_region(region_path: Path) -> dict:
    """Parse box region from DS9 region file.

    Returns dict with: xc, yc, width, height, angle
    """
    with open(region_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("box"):
                # Parse box(xc, yc, width, height, angle)
                params = line[line.find("(") + 1:line.find(")")].split(",")
                return {
                    "xc": float(params[0]),
                    "yc": float(params[1]),
                    "width": float(params[2]),
                    "height": float(params[3]),
                    "angle": float(params[4]) if len(params) > 4 else 0.0,
                }

    raise ValueError(f"No box region found in {region_path}")


def compute_rotated_box_center(
    box: dict, image_center: tuple[float, float]
) -> tuple[float, float]:
    """Compute where box center lands after rotating image to align box."""
    cx, cy = image_center
    xc, yc = box["xc"], box["yc"]
    angle_rad = -np.radians(box["angle"])

    # Translate to origin, rotate, translate back
    dx, dy = xc - cx, yc - cy
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    new_x = dx * cos_a - dy * sin_a + cx
    new_y = dx * sin_a + dy * cos_a + cy

    return new_x, new_y


def rotate_image(
    data: np.ndarray,
    angle: float,
    order: int = 3
) -> np.ndarray:
    """Rotate image by angle (degrees) around center."""
    if angle == 0:
        return data.copy()
    return rotate(data, angle, reshape=False, order=order, mode="constant", cval=0)


def update_header_for_rotation(header: fits.Header, angle: float) -> fits.Header:
    """Update WCS in header for rotation around image center."""
    header = header.copy()

    if angle == 0:
        return header

    # Check if WCS exists
    if "CRPIX1" not in header:
        return header

    try:
        wcs = WCS(header, naxis=2)

        # Rotation matrix
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Update CD matrix or CDELT/CROTA
        if "CD1_1" in header:
            cd = np.array([
                [header.get("CD1_1", 1), header.get("CD1_2", 0)],
                [header.get("CD2_1", 0), header.get("CD2_2", 1)]
            ])
            new_cd = rot_matrix @ cd
            header["CD1_1"] = new_cd[0, 0]
            header["CD1_2"] = new_cd[0, 1]
            header["CD2_1"] = new_cd[1, 0]
            header["CD2_2"] = new_cd[1, 1]
        elif "CDELT1" in header:
            # Simple case: just update CROTA if present
            if "CROTA2" in header:
                header["CROTA2"] = header["CROTA2"] + angle
            elif "CROTA1" in header:
                header["CROTA1"] = header["CROTA1"] + angle

    except Exception:
        # If WCS update fails, just return original header
        pass

    return header


def update_header_for_cutout(
    header: fits.Header,
    x_min: int,
    y_min: int,
    new_shape: tuple[int, int]
) -> fits.Header:
    """Update header for cutout region."""
    header = header.copy()

    # Update image dimensions
    header["NAXIS1"] = new_shape[1]
    header["NAXIS2"] = new_shape[0]

    # Update reference pixel (CRPIX) for WCS
    if "CRPIX1" in header:
        header["CRPIX1"] = header["CRPIX1"] - x_min
    if "CRPIX2" in header:
        header["CRPIX2"] = header["CRPIX2"] - y_min

    # Add cutout info to header
    header["CUTOUT"] = True
    header["CUT_XMIN"] = x_min
    header["CUT_YMIN"] = y_min

    return header


def process_science_or_sigma(
    input_path: Path,
    output_path: Path,
    box: dict,
    image_center: tuple[float, float],
    is_inverse_variance: bool = False,
    overwrite: bool = False
) -> None:
    """Rotate and cut a science or sigma image."""
    with fits.open(input_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header.copy()

    angle = box["angle"]

    # Rotate image
    rotated = rotate_image(data, angle)
    header = update_header_for_rotation(header, angle)

    # Compute cutout bounds (box center after rotation)
    new_xc, new_yc = compute_rotated_box_center(box, image_center)
    half_w, half_h = box["width"] / 2, box["height"] / 2

    x_min = int(np.floor(new_xc - half_w))
    x_max = int(np.ceil(new_xc + half_w))
    y_min = int(np.floor(new_yc - half_h))
    y_max = int(np.ceil(new_yc + half_h))

    # Bounds checking
    y_max_img, x_max_img = rotated.shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(x_max_img, x_max)
    y_max = min(y_max_img, y_max)

    # Cut
    cutout = rotated[y_min:y_max, x_min:x_max]

    # Convert inverse variance to sigma if needed
    if is_inverse_variance:
        if np.any(cutout <= 0):
            # Replace non-positive values with nan, then convert
            cutout = np.where(cutout > 0, cutout, np.nan)
            print(f"[WARN] {input_path.name}: replaced non-positive inverse variance with NaN")
        cutout = 1.0 / np.sqrt(cutout)

    # Update header
    header = update_header_for_cutout(header, x_min, y_min, cutout.shape)

    # Write
    hdu = fits.PrimaryHDU(cutout.astype(np.float32), header=header)
    hdu.writeto(output_path, overwrite=overwrite)
    print(f"Written: {output_path}")


def process_psf(
    input_path: Path,
    output_path: Path,
    angle: float,
    overwrite: bool = False
) -> None:
    """Rotate a PSF image (no cutting)."""
    with fits.open(input_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header.copy()

    # Rotate with higher order interpolation for PSF
    rotated = rotate_image(data, angle, order=3)

    # Preserve flux normalization
    if np.sum(data) > 0:
        rotated *= np.sum(data) / np.sum(rotated)

    # Update header
    header = update_header_for_rotation(header, angle)
    header["PSF_ROT"] = angle
    header.add_history(f"Rotated by {angle:.4f} degrees for cutout alignment")

    # Write
    hdu = fits.PrimaryHDU(rotated.astype(np.float32), header=header)
    hdu.writeto(output_path, overwrite=overwrite)
    print(f"Written: {output_path}")


def resolve_output_path(input_path: Path, prefix: str, force: bool) -> Path:
    """Generate output path with prefix, handling versioning."""
    directory = input_path.parent
    # Avoid double-prefixing if input already has the prefix
    if input_path.name.startswith(prefix):
        output_name = input_path.name
    else:
        output_name = f"{prefix}{input_path.name}"
    output_path = directory / output_name

    if force or not output_path.exists():
        return output_path

    # Version if exists
    stem = input_path.stem
    suffix = input_path.suffix
    i = 1
    while True:
        versioned = directory / f"{prefix}{stem}_v{i}{suffix}"
        if not versioned.exists():
            return versioned
        i += 1


def main():
    args = parse_args()

    region_path = Path(args.region)

    # Need at least some input
    if not args.science and not args.sigma and not args.psf:
        sys.exit("[ERROR] No input files. Use --science, --sigma, or --psf")

    # Parse box region
    box = parse_box_region(region_path)
    print(f"Box: center=({box['xc']:.1f}, {box['yc']:.1f}), "
          f"size={box['width']:.0f}x{box['height']:.0f}, angle={box['angle']:.1f}°")

    # Determine image center
    if args.science:
        ref_file = Path(args.science[0])
    elif args.sigma:
        ref_file = Path(args.sigma[0])
    elif args.reference:
        ref_file = Path(args.reference)
    else:
        sys.exit("[ERROR] Need --reference when only rotating PSF files")

    image_center = get_image_center(ref_file)
    print(f"Image center: ({image_center[0]:.1f}, {image_center[1]:.1f})")

    # Process science files
    for f in args.science:
        input_path = Path(f)
        output_path = resolve_output_path(input_path, args.prefix, args.force)
        process_science_or_sigma(
            input_path, output_path, box, image_center,
            is_inverse_variance=False, overwrite=args.force
        )

    # Process sigma files
    for f in args.sigma:
        input_path = Path(f)
        output_path = resolve_output_path(input_path, args.prefix, args.force)
        process_science_or_sigma(
            input_path, output_path, box, image_center,
            is_inverse_variance=args.inverse_variance, overwrite=args.force
        )

    # Process PSF files (rotate only, no cut)
    for f in args.psf:
        input_path = Path(f)
        output_path = resolve_output_path(input_path, args.prefix, args.force)
        process_psf(input_path, output_path, box["angle"], overwrite=args.force)

    print("Done.")


if __name__ == "__main__":
    main()
