from pathlib import Path
import argparse

from astropy.io import fits

from galfit_builder.config.loader import load_config
from galfit_builder.controls.controls import GalfitControls
from galfit_builder.io.parse_regions import read_regions
from galfit_builder.builders.component_builder import build_components
from galfit_builder.masking.region_mask import build_region_mask
from galfit_builder.constraints.constraint_builder import build_constraints


def _parse_fit_region(value, image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Parse fit_region from config: "auto" uses full image, list/tuple used as-is."""
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        ny, nx = image_shape
        return (1, nx, 1, ny)
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(value)
    raise ValueError(f"fit_region must be 'auto' or [xmin, xmax, ymin, ymax], got: {value}")


def _parse_conv_box(value, psf_path: str | None) -> tuple[int, int]:
    """Parse conv_box from config: "auto" uses PSF dimensions, list/tuple used as-is."""
    if isinstance(value, str) and value.lower() == "auto":
        if psf_path and psf_path.lower() not in ("none", "auto"):
            with fits.open(psf_path) as hdul:
                ny, nx = hdul[0].data.shape
                return (nx, ny)
        return (100, 100)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return tuple(value)
    raise ValueError(f"conv_box must be 'auto' or [x, y], got: {value}")


def _parse_plate_scale(value, header) -> tuple[float, float]:
    """Parse plate_scale from config: "auto" reads from FITS header, list/tuple used as-is."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return tuple(value)
    if isinstance(value, str) and value.lower() == "auto":
        if "CDELT1" in header and "CDELT2" in header:
            return (abs(header["CDELT1"]) * 3600, abs(header["CDELT2"]) * 3600)
        if "CD1_1" in header:
            ps1 = abs(header.get("CD1_1", 0)) + abs(header.get("CD1_2", 0))
            ps2 = abs(header.get("CD2_1", 0)) + abs(header.get("CD2_2", 0))
            return (ps1 * 3600, ps2 * 3600)
        raise ValueError("plate_scale='auto' but FITS header has no CDELT or CD matrix; provide explicit [dx, dy]")
    raise ValueError(f"plate_scale must be 'auto' or [dx, dy], got: {value}")


def _parse_zeropoint(value, header) -> float:
    """Parse zeropoint from config: "auto" reads from FITS header, number used as-is."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.lower() == "auto":
        for key in ("ZEROPNT", "ZEROPOINT", "MAGZPT", "PHOTZPT", "ZP"):
            if key in header:
                return float(header[key])
        raise ValueError("zeropoint='auto' but FITS header has no zeropoint keyword; provide explicit value")
    return float(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Create GALFIT feedme file from TOML config file")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("-c", "--constraints", action="store_true",
                        help="Generate constraint file from config")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force overwrite existing files")

    return parser.parse_args()


def resolve_output_path(directory: Path, stem: str, suffix: str,
                        version_outputs: bool, overwrite: bool) -> Path:
    """Resolve output path, optionally adding version numbers to avoid overwrites.

    First run uses plain name (stem.suffix), subsequent runs use _v1, _v2, etc.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    base_path = directory / f"{stem}{suffix}"

    if overwrite:
        return base_path

    if not version_outputs:
        return base_path

    # First run: use plain name if it doesn't exist
    if not base_path.exists():
        return base_path

    # Subsequent runs: find next available version
    i = 1
    while True:
        path = directory / f"{stem}_v{i}{suffix}"
        if not path.exists():
            return path
        i += 1


def main():
    args = parse_args()

    config = load_config(args.config)
    control_inputs = config["galfit_input_controls"]
    working_dir = Path(control_inputs["working_dir"])

    control_outputs = config["galfit_output_controls"]
    version_outputs = control_outputs["version_outputs"]
    overwrite = control_outputs["overwrite"] or args.force

    # Derive output stem from input filename (e.g., "179485_F115W_f" from "179485_F115W_f.fits")
    input_stem = Path(control_inputs["input_data_image"]).stem

    output_data_image = resolve_output_path(
        working_dir, f"{input_stem}_output", ".fits",
        version_outputs, overwrite
    )

    feedme_file = resolve_output_path(
        working_dir, f"{input_stem}_galfit", ".feedme",
        version_outputs, overwrite
    )

    # Read regions and image data/header first
    regions = read_regions(config["input"]["region_file"])
    with fits.open(control_inputs["input_data_image"]) as hdul:
        image_data = hdul[0].data
        image_header = hdul[0].header

    # Determine mask: user-provided, auto-generated from exclusions, or none
    user_mask = control_inputs.get("mask_image", "auto")
    if user_mask.lower() not in ("auto", "none"):
        # User provided their own mask
        mask_image = user_mask
    else:
        # Try to generate from excluded regions
        mask_output = resolve_output_path(
            working_dir, f"{input_stem}_mask", ".fits",
            version_outputs, overwrite
        )
        mask_generated = build_region_mask(regions, image_data.shape, mask_output)
        mask_image = mask_output if mask_generated else "none"

    # Build components first (needed for constraint generation)
    components = build_components(regions, image_data, config)

    # Determine constraint file: user-provided, auto-generated, or none
    user_constraint = control_inputs.get("constraint_file", "auto")
    if user_constraint.lower() not in ("auto", "none"):
        constraint_file = user_constraint
    elif args.constraints:
        constraint_output = resolve_output_path(
            working_dir, f"{input_stem}_constraints", ".txt",
            version_outputs, overwrite
        )
        constraint_generated = build_constraints(components, config, constraint_output)
        constraint_file = constraint_output if constraint_generated else "none"
    else:
        constraint_file = "none"

    psf_image = control_inputs.get("psf_image", "auto")

    controls = GalfitControls(
        working_dir=working_dir,
        input_data_image=control_inputs["input_data_image"],
        output_data_image=output_data_image,
        zeropoint=_parse_zeropoint(control_inputs.get("zeropoint", "auto"), image_header),
        plate_scale=_parse_plate_scale(control_inputs.get("plate_scale", "auto"), image_header),
        sigma_image=control_inputs.get("sigma_image", "auto"),
        psf_image=psf_image,
        mask_image=mask_image,
        constraint_file=constraint_file,
        psf_fine_samp_factor=control_inputs["psf_sampling"],
        fit_region=_parse_fit_region(control_inputs.get("fit_region", "auto"), image_data.shape),
        conv_box_size=_parse_conv_box(control_inputs.get("conv_box", "auto"), psf_image),
        display_type=control_inputs["display_type"],
        mode=control_inputs["mode"],
    )

    controls.validate_inputs()

    with open(feedme_file, "w") as f:
        f.write(str(controls))
        for comp in components:
            f.write(str(comp))
            f.write("\n")
