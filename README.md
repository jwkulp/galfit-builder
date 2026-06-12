# galfit-builder

A Python CLI toolkit for working with GALFIT galaxy fitting software. Convert DS9 region files to GALFIT feedme files, freeze components, cut/rotate images, and convert results back to regions.

## Installation

```bash
pip install galfit-builder
```

For development:
```bash
git clone https://github.com/YOUR_USERNAME/galfit-builder.git
cd galfit-builder
pip install -e ".[dev]"
```

## CLI Tools

### galfit-builder

Convert DS9 regions to a GALFIT feedme file.

```bash
galfit-builder config.toml                    # basic usage
galfit-builder config.toml -c                 # also generate constraints file
galfit-builder config.toml -o custom.feedme   # custom output name
```

### galfit-freeze

Freeze components outside green polygon regions. Useful for iterative fitting — lock down fitted components while working on a specific area.

```bash
galfit-freeze output.feedme regions.reg
galfit-freeze output.feedme regions.reg -i image.fits -c config.toml  # add new components
galfit-freeze output.feedme regions.reg -o frozen.feedme
```

### galfit-cutout

Rotate and cut FITS images based on a DS9 box region. Smaller cutouts run faster in GALFIT.

```bash
galfit-cutout cutout.reg -s science.fits -e sigma.fits -p psf.fits
galfit-cutout cutout.reg -s image.fits --prefix rot_     # custom prefix
galfit-cutout cutout.reg -s image.fits -v                # sigma is inverse variance
galfit-cutout cutout.reg -p psf.fits -r reference.fits   # PSF only, need reference for center
```

### galfit-to-regions

Convert a GALFIT feedme (or output) back to DS9 regions. Useful for visualizing fitted positions after a GALFIT run.

```bash
galfit-to-regions config.toml output.feedme
galfit-to-regions config.toml galfit.01 -o fitted.reg
```

## Typical Workflow

1. Create a config file (copy from `src/galfit_builder/config/example_config.toml`)
2. Draw regions in DS9, save as `regions.reg`
3. `galfit-builder config.toml` → creates feedme
4. Run GALFIT: `galfit feedme.galfit`
5. `galfit-to-regions config.toml galfit.01` → visualize fitted positions in DS9
6. Adjust regions, use `galfit-freeze` to lock components you're happy with
7. Repeat until satisfied

## Configuration

See `src/galfit_builder/config/example_config.toml` for a fully documented configuration file.

### Region Color Mapping

DS9 region colors map to GALFIT component types:

| Color   | Component |
|---------|-----------|
| green   | sersic    |
| red     | devauc    |
| blue    | ferrer    |
| cyan    | gaussian  |
| yellow  | moffat    |
| magenta | king      |
| white   | nuker     |
| orange  | expdisk   |
| pink    | edgedisk  |

Point regions are always PSF components regardless of color.

### The "auto" Keyword

Use `"auto"` in config for smart defaults:
- **Optional files** (sigma, mask, constraint): writes `"none"` to GALFIT
- **Numeric values** (sky background, magnitudes): computed from image data

## Supported Components

All 11 GALFIT component types:

| Component | Description | Region Shape |
|-----------|-------------|--------------|
| sersic | Sersic profile | Ellipse |
| psf | Point spread function | Point |
| devauc | de Vaucouleurs (n=4) | Ellipse |
| expdisk | Exponential disk (n=1) | Ellipse |
| ferrer | Ferrer profile | Ellipse |
| gaussian | 2D Gaussian | Ellipse |
| moffat | Moffat profile | Ellipse |
| king | King profile (globular clusters) | Ellipse |
| nuker | Nuker profile (galaxy nuclei) | Ellipse |
| edgedisk | Edge-on disk | Ellipse |
| sky | Sky background | N/A (config or auto-estimated) |

## Python API

```python
from galfit_builder.config.loader import load_config
from galfit_builder.regions.parse_regions import read_regions
from galfit_builder.builders.component_builder import build_components
from galfit_builder.controls.controls import GalfitControls
from astropy.io import fits

# Load configuration
config = load_config("config.toml")

# Read regions and image data
regions = read_regions("sources.reg")
image_data = fits.getdata("input.fits")

# Build components from regions
components = build_components(regions, image_data, config, add_sky=True)

# Generate feedme output
controls = GalfitControls.from_config(config)
print(controls.to_galfit())
for comp in components:
    print(comp.to_galfit())
```

## Publishing to GitHub

```bash
# Initialize git (if not already)
cd galfit-builder
git init
git add .
git commit -m "Initial commit"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/galfit-builder.git
git branch -M main
git push -u origin main
```

## Publishing to PyPI

1. **Create accounts** at [pypi.org](https://pypi.org) and optionally [test.pypi.org](https://test.pypi.org)

2. **Install build tools**:
```bash
pip install build twine
```

3. **Build the package**:
```bash
python -m build
```
This creates `dist/galfit_builder-0.1.0.tar.gz` and `dist/galfit_builder-0.1.0-py3-none-any.whl`

4. **Test on TestPyPI first** (optional but recommended):
```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ galfit-builder
```

5. **Upload to PyPI**:
```bash
twine upload dist/*
```

6. **For future releases**, bump the version in `pyproject.toml` and repeat steps 3-5.

### Using API Tokens (Recommended)

Instead of username/password, use API tokens:

1. Go to PyPI → Account Settings → API tokens
2. Create a token scoped to this project
3. Use with twine:
```bash
twine upload dist/* -u __token__ -p pypi-YOUR_TOKEN_HERE
```

Or create `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE
```

## Development

```bash
pytest                          # run tests
mypy src/galfit_builder         # type checking
ruff check src/galfit_builder   # linting
```

## License

MIT
