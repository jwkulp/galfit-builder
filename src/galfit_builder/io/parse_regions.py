from __future__ import annotations
from pathlib import Path
from regions import Regions, PixelRegion


def read_regions(path: str | Path) -> list[PixelRegion]:
    """Read DS9 region file and return list of pixel regions."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Region file not found: {path}")

    return list(Regions.read(path, format="ds9"))
