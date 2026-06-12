from pathlib import Path
Pathish = str | Path
import os


def _normalize_optional_path(value: Pathish | None) -> Path | None:
    """Convert 'auto' or 'none' strings to None, otherwise return Path."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("auto", "none"):
        return None
    return Path(value)


class GalfitControls:
    def __init__(
        self,
        *,
        working_dir: Path,
        input_data_image: Pathish,
        output_data_image: Pathish,
        zeropoint: float,
        plate_scale: tuple[float, float],

        sigma_image: Pathish | None = None,
        psf_image: Pathish | None = None,
        mask_image: Pathish | None = None,
        constraint_file: Pathish | None = None,

        psf_fine_samp_factor: int = 1,
        fit_region: tuple[int, int, int, int] | None = None,
        conv_box_size: tuple[int, int] | None = None,
        display_type: str = "regular",
        mode: int = 0
    ):

        self.working_dir = Path(working_dir).resolve()
        if not self.working_dir.exists():
            raise FileNotFoundError(f"Working directory not found: {self.working_dir}")

        self.input_data_image = Path(input_data_image)
        self.output_data_image = Path(output_data_image)
        self.zeropoint = zeropoint
        self.plate_scale = plate_scale
        self.sigma_image = _normalize_optional_path(sigma_image)
        self.psf_image = _normalize_optional_path(psf_image)
        self.mask_image = _normalize_optional_path(mask_image)
        self.constraint_file = _normalize_optional_path(constraint_file)
        self.psf_fine_samp_factor = psf_fine_samp_factor
        self.fit_region = fit_region
        self.conv_box_size = conv_box_size
        self.display_type = display_type
        self.mode = mode


    def _feedme_path(self, path: Path | None):
        if path is None:
            return "none"

        rel = os.path.relpath(path, start=self.working_dir)

        if len(rel) >= 80:
            raise ValueError(f"Relative path too long for GALFIT (must be < 80): {rel}")

        return rel


    def _check_exists(self, path: Path | None):
        if path is None:
            return
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")


    def validate_inputs(self):
        self._check_exists(self.input_data_image)
        self._check_exists(self.psf_image)
        self._check_exists(self.sigma_image)
        self._check_exists(self.mask_image)
        self._check_exists(self.constraint_file)

        if self.zeropoint <= 0:
            raise ValueError("Zeropoint must be positive")

        if any(ps <= 0 for ps in self.plate_scale):
            raise ValueError("Plate scale values must be positive")

        if self.psf_fine_samp_factor < 1:
            raise ValueError("PSF fine sampling factor must be >= 1")

        if self.fit_region is not None:
            xmin, xmax, ymin, ymax = self.fit_region
            if xmin > xmax or ymin > ymax:
                raise ValueError("Invalid fitting region")

        if self.conv_box_size is not None:
            x, y = self.conv_box_size
            if x <= 0 or y <= 0:
                raise ValueError("Invalid convolution box")

        if self.mode not in range(0,4):
            raise ValueError("Invalid mode input")

        _ALLOWED_DISPLAY_TYPES = {"regular", "curses", "both"}
        if self.display_type.lower() not in _ALLOWED_DISPLAY_TYPES:
            raise ValueError(f"Invalid display type: must be one of {_ALLOWED_DISPLAY_TYPES}")


    def __str__(self):
        lines = []

        lines.append("# IMAGE PARAMETERS")
        lines.append(f"A) {self._feedme_path(self.input_data_image)}            # Input data image (FITS file)")
        lines.append(f"B) {self._feedme_path(self.output_data_image)}           # Output data image block")
        lines.append(f'C) {self._feedme_path(self.sigma_image)}                 # Sigma image name (made from data if blank or "none")')
        lines.append(f"D) {self._feedme_path(self.psf_image)}                   # Input PSF image and (optional) diffusion kernel")
        lines.append(f"E) {self.psf_fine_samp_factor}                           # PSF fine sampling factor relative to data")
        lines.append(f"F) {self._feedme_path(self.mask_image)}                  # Bad pixel mask (FITS image or ASCII coord list)")
        lines.append(f"G) {self._feedme_path(self.constraint_file)}             # File with parameter constraints (ASCII file)")

        if self.fit_region is not None:
            xmin, xmax, ymin, ymax = self.fit_region
            lines.append(f"H) {xmin} {xmax} {ymin} {ymax}             # Image region to fit (xmin xmax ymin ymax)")
        else:
            lines.append("H) none                     # Image region to fit (xmin xmax ymin ymax)")

        if self.conv_box_size is not None:
            x, y = self.conv_box_size
            lines.append(f"I) {x} {y}                   # Size of the convolution box (x y)")
        else:
            lines.append("I) none                     # Size of the convolution box (x y)")

        lines.append(f"J) {self.zeropoint}                    # Magnitude photometric zeropoint")

        dx, dy = self.plate_scale
        lines.append(f"K) {dx} {dy}               # Plate scale (dx dy) [arcsec per pixel]")

        lines.append(f"O) {self.display_type}                  # Display type (regular, curses, both)")
        lines.append(f"P) {self.mode}                           # Options: 0=normal run; 1,2=make model/imgblock & quit")

        lines.append("")
        lines.append("# INITIAL FITTING PARAMETERS")
        lines.append("#")
        lines.append("# For object type, the allowed functions are:")
        lines.append("#   nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat,")
        lines.append("#   ferrer, and sky.")
        lines.append("#")
        lines.append("# Hidden parameters will only appear when they're specified:")
        lines.append("#   C0 (diskyness/boxyness),")
        lines.append("#   Fn (n=integer, Azimuthal Fourier Modes).")
        lines.append("#   R0-R10 (PA rotation, for creating spiral structures).")
        lines.append("#")
        lines.append("# " + "-" * 78)
        lines.append("#   par)    par value(s)    fit toggle(s)    # parameter description")
        lines.append("# " + "-" * 78)

        final_str = "\n".join(lines) + "\n"
        return final_str
