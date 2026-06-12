import numpy as np
from astropy.io import fits
from pathlib import Path


def build_region_mask(regions, shape, mask_file):
    """Build a mask from excluded regions and write to FITS file.

    Returns True if a mask was written, False if no exclusions found.
    """
    mask = np.zeros(shape, dtype=np.uint8)

    exclusion_found = False
    for region in regions:
        if region.meta.get("include", 1) == 0:
            region_mask = region.to_mask(mode="center")
            region_mask_image = region_mask.to_image(shape)

            if region_mask_image is not None:
                mask[region_mask_image > 0] = 1
                exclusion_found = True

    if exclusion_found and np.any(mask):
        fits.PrimaryHDU(mask).writeto(mask_file, overwrite=True)
        return True

    return False
