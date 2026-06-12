from pathlib import Path
import argparse

from astropy.io import fits

from galfit_builder.config.loader import load_config
from galfit_builder.controls.controls import GalfitControls
from galfit_builder.io.parse_regions import read_regions
from galfit_builder.builders.component_builder import build_components
from galfit_builder.masking.region_mask import build_region_mask
from galfit_builder.constraints.constraint_builder import build_constraints


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

    # Read regions and image data first
    regions = read_regions(config["input"]["region_file"])
    image_data = fits.getdata(control_inputs["input_data_image"])

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

    controls = GalfitControls(
        working_dir=working_dir,
        input_data_image=control_inputs["input_data_image"],
        output_data_image=output_data_image,
        zeropoint=control_inputs["zeropoint"],
        plate_scale=control_inputs["plate_scale"],
        sigma_image=control_inputs.get("sigma_image", "auto"),
        psf_image=control_inputs.get("psf_image", "auto"),
        mask_image=mask_image,
        constraint_file=constraint_file,
        psf_fine_samp_factor=control_inputs["psf_sampling"],
        fit_region=control_inputs["fit_region"],
        conv_box_size=control_inputs["conv_box"],
        display_type=control_inputs["display_type"],
        mode=control_inputs["mode"],
    )

    controls.validate_inputs()

    with open(feedme_file, "w") as f:
        f.write(str(controls))
        for comp in components:
            f.write(str(comp))
            f.write("\n")
