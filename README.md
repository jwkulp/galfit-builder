# galfit-builder

A Python CLI toolkit for GALFIT galaxy fitting. Convert DS9 region files to GALFIT feedme files, freeze fitted components, cut/rotate images, and convert results back to regions for visualization.

## Installation

```bash
pip install galfit-builder
```

## Quick Start

1. Copy the example config and edit for your data:
```bash
cp $(python -c "import galfit_builder; print(galfit_builder.__path__[0])")/config/example_config.toml config.toml
```

2. Draw regions in DS9, save as `regions.reg`

3. Generate feedme:
```bash
galfit-builder config.toml
```

4. Run GALFIT:
```bash
galfit output_galfit.feedme
```

5. Visualize results:
```bash
galfit-to-regions config.toml galfit.01
```

## CLI Tools

### galfit-builder

Convert DS9 regions to a GALFIT feedme file.

```bash
galfit-builder config.toml                    # generate feedme
galfit-builder config.toml -c                 # also generate constraints file
galfit-builder config.toml -o custom.feedme   # custom output name
```

### galfit-freeze

Freeze components outside green polygon regions. Lock down fitted components while working on a specific area.

```bash
galfit-freeze output.feedme regions.reg                              # freeze only
galfit-freeze output.feedme regions.reg -i image.fits -c config.toml # add new components
```

Draw a green polygon in DS9 around the area you want to fit. Components inside remain free; components outside are frozen.

### galfit-cutout

Rotate and cut FITS images based on a DS9 box region. Smaller cutouts run faster in GALFIT.

```bash
galfit-cutout box.reg -s science.fits -e sigma.fits -p psf.fits
galfit-cutout box.reg -s science.fits -v              # sigma is inverse variance
galfit-cutout box.reg -s science.fits --prefix rot_   # custom output prefix
```

### galfit-to-regions

Convert GALFIT output back to DS9 regions for visualization.

```bash
galfit-to-regions config.toml galfit.01           # GALFIT output file
galfit-to-regions config.toml output.feedme       # or feedme file
```

## Configuration

### Minimal Config

```toml
[input]
region_file = "regions.reg"

[galfit_input_controls]
working_dir = "."
input_data_image = "science.fits"
psf_image = "psf.fits"
zeropoint = 25.0
plate_scale = [0.03, 0.03]
fit_region = [1, 500, 1, 500]
conv_box = [100, 100]
display_type = "regular"
mode = 0

[galfit_output_controls]
version_outputs = true
overwrite = false

[region_colors]
green = "sersic"
red = "devauc"
cyan = "gaussian"
```

### Region Color Mapping

DS9 region colors determine GALFIT component types:

| Color   | Component | Description |
|---------|-----------|-------------|
| green   | sersic    | Sersic profile |
| red     | devauc    | de Vaucouleurs (n=4) |
| blue    | ferrer    | Ferrer profile |
| cyan    | gaussian  | 2D Gaussian |
| yellow  | moffat    | Moffat profile |
| magenta | king      | King profile |
| white   | nuker     | Nuker profile |
| orange  | expdisk   | Exponential disk |
| pink    | edgedisk  | Edge-on disk |

Point regions become PSF components regardless of color.

### The "auto" Keyword

Use `"auto"` for smart defaults:

```toml
sigma_image = "auto"    # writes "none" — let GALFIT estimate
background = "auto"     # compute sky from image edges
```

### Sky Component

Sky is added automatically as component 1. Disable with:

```toml
[defaults.sky]
include = false
```

Or configure:

```toml
[defaults.sky]
include = true
background = "auto"       # or a number like 100.5
dsky_dx = 0.0
dsky_dy = 0.0
background_freeze = false
dsky_dx_freeze = true
dsky_dy_freeze = true
```

### Constraints

Generate parameter constraints with `-c`:

```bash
galfit-builder config.toml -c
```

Configure in TOML:

```toml
[constraints.sersic]
n = "0.5 to 8"           # hard limits
q = "0.1 to 1"
re_scale = 0.1           # re constrained to [re*0.1, re/0.1]
mag = "10 to 35"
x = "-5 5"               # offset from initial: +/- 5 pixels
y = "-5 5"
```

### Component Defaults

Set default parameters and freeze states:

```toml
[defaults.sersic]
sersic_index = 2.5
mag_freeze = false
sersic_index_freeze = false
axis_ratio_freeze = false
pos_angle_freeze = false

[defaults.psf]
mag_freeze = false
```

## Workflow Example

**Initial fit:**
```bash
# Draw ellipses/points in DS9, save as regions.reg
galfit-builder config.toml
galfit output_galfit.feedme
galfit-to-regions config.toml galfit.01
# Load galfit.01.reg in DS9 to see fitted positions
```

**Iterative refinement:**
```bash
# Draw green polygon around area to refine
# Adjust regions inside polygon
galfit-freeze galfit.01 regions.reg -i science.fits -c config.toml
galfit frozen_galfit.01.feedme
```

**Working with cutouts:**
```bash
# Draw box region in DS9 for cutout area
galfit-cutout box.reg -s science.fits -e sigma.fits -p psf.fits
# Edit config to point to cutout_* files
galfit-builder config.toml
```

## Python API

```python
from galfit_builder.config.loader import load_config
from galfit_builder.io.parse_regions import read_regions
from galfit_builder.builders.component_builder import build_components
from galfit_builder.controls.controls import GalfitControls
from astropy.io import fits

config = load_config("config.toml")
regions = read_regions("regions.reg")
image_data = fits.getdata("science.fits")

components = build_components(regions, image_data, config)

controls = GalfitControls.from_config(config)
print(controls.to_galfit())
for comp in components:
    print(comp.to_galfit())
```

## Supported Components

| Component | Region | Key Parameters |
|-----------|--------|----------------|
| sersic | Ellipse | magnitude, effective radius, sersic index, axis ratio, PA |
| psf | Point | magnitude |
| devauc | Ellipse | magnitude, effective radius, axis ratio, PA |
| expdisk | Ellipse | magnitude, scale radius, axis ratio, PA |
| gaussian | Ellipse | magnitude, FWHM, axis ratio, PA |
| moffat | Ellipse | magnitude, FWHM, powerlaw, axis ratio, PA |
| king | Ellipse | surface brightness, core radius, tidal radius, alpha |
| nuker | Ellipse | surface brightness, break radius, alpha, beta, gamma |
| ferrer | Ellipse | surface brightness, outer radius, alpha, beta |
| edgedisk | Ellipse | surface brightness, scale height, scale length, PA |
| sky | N/A | background, dsky/dx, dsky/dy |

## License

MIT
