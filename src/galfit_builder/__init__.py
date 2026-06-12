"""
galfit_builder - Convert DS9 region files to GALFIT feedme files.

Usage:
    from galfit_builder import build_components, load_config, GalfitControls

    config = load_config("config.toml")
    components = build_components(regions, image_data, config)

    # Optionally add sky
    components = build_components(regions, image_data, config, add_sky=True)
"""

from galfit_builder.config.loader import load_config
from galfit_builder.controls.controls import GalfitControls

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy import for modules that require optional dependencies."""
    if name == "build_components":
        from galfit_builder.builders.component_builder import build_components
        return build_components
    if name == "read_regions":
        from galfit_builder.io.parse_regions import read_regions
        return read_regions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_components",
    "load_config",
    "GalfitControls",
    "read_regions",
]
