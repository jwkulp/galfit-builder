"""Convert GALFIT feedme file back to DS9 region file.

Useful after a GALFIT run completes — extract fitted positions and shapes
back into regions for visualization or further iteration.
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Any

from galfit_builder.config.loader import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GALFIT feedme file to DS9 region file"
    )
    parser.add_argument("config", help="Config file (for color mapping)")
    parser.add_argument("feedme", help="GALFIT feedme file to convert")
    parser.add_argument("--output", "-o", help="Output region file (default: <feedme>.reg)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite existing files")
    return parser.parse_args()


def get_color_map(config: dict[str, Any]) -> dict[str, str]:
    """Build component type -> color map from config (reversed from config's color -> type)."""
    return {v: k for k, v in config["region_colors"].items()}


def parse_param_line(line: str) -> tuple[str, list[str]]:
    """Parse a GALFIT parameter line like ' 1) 100.5 200.3 1 1  # comment'.

    Returns (param_number, values_list)
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return "", []

    match = re.match(r'^(\d+)\)\s+(.+?)(?:\s*#.*)?$', line)
    if not match:
        return "", []

    param_num = match.group(1)
    values_str = match.group(2).strip()
    values = values_str.split()

    return param_num, values


def parse_components(lines: list[str]) -> list[dict[str, Any]]:
    """Parse all components from feedme lines."""
    components: list[dict[str, Any]] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("0)"):
            parts = line.split()
            if len(parts) >= 2:
                comp_type = parts[1].lower()

                if comp_type == "sky":
                    i += 1
                    continue

                comp: dict[str, Any] = {"type": comp_type}

                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("0)"):
                    param_line = lines[j].strip()

                    if param_line.startswith("# Object") or param_line.startswith("# Component"):
                        break

                    param_num, values = parse_param_line(param_line)
                    if param_num and values:
                        comp[f"param_{param_num}"] = values

                    j += 1

                if "param_1" in comp and len(comp["param_1"]) >= 2:
                    comp["x"] = float(comp["param_1"][0])
                    comp["y"] = float(comp["param_1"][1])
                    components.append(comp)

                i = j
                continue

        i += 1

    return components


def get_ellipse_params(comp: dict[str, Any]) -> tuple[float, float, float] | None:
    """Extract ellipse parameters from component.

    Returns (semi_major, semi_minor, position_angle) or None if not applicable.
    """
    comp_type = comp["type"]
    size: float | None = None

    if comp_type in ("sersic", "devauc"):
        if "param_4" in comp:
            size = float(comp["param_4"][0])

    elif comp_type == "expdisk":
        if "param_4" in comp:
            size = float(comp["param_4"][0])

    elif comp_type in ("gaussian", "moffat"):
        if "param_4" in comp:
            size = float(comp["param_4"][0]) / 2.0

    elif comp_type == "king":
        if "param_5" in comp:
            size = float(comp["param_5"][0])

    elif comp_type == "nuker":
        if "param_4" in comp:
            size = float(comp["param_4"][0])

    elif comp_type == "ferrer":
        if "param_4" in comp:
            size = float(comp["param_4"][0])

    elif comp_type == "edgedisk":
        if "param_5" in comp and "param_4" in comp:
            length = float(comp["param_5"][0])
            height = float(comp["param_4"][0])
            pa = float(comp["param_10"][0]) if "param_10" in comp else 0.0
            ds9_angle = (pa + 90) % 360
            return length, height, ds9_angle

    if size is None:
        return None

    q = float(comp["param_9"][0]) if "param_9" in comp else 1.0
    pa = float(comp["param_10"][0]) if "param_10" in comp else 0.0

    a = size
    b = size * q
    ds9_angle = (pa + 90) % 360

    return a, b, ds9_angle


def component_to_region(comp: dict[str, Any], color_map: dict[str, str]) -> str | None:
    """Convert a component to a DS9 region string."""
    comp_type = comp["type"]
    x: float = comp["x"]
    y: float = comp["y"]
    color = color_map.get(comp_type, "green")

    if comp_type == "psf":
        return f"point({x:.4f},{y:.4f}) # point=circle color={color}"

    ellipse_params = get_ellipse_params(comp)
    if ellipse_params:
        a, b, angle = ellipse_params
        return f"ellipse({x:.4f},{y:.4f},{a:.4f},{b:.4f},{angle:.4f}) # color={color}"

    return None


def resolve_output_path(input_path: Path, output: str | None, force: bool) -> Path:
    """Determine output path."""
    if output:
        return Path(output)

    directory = input_path.parent
    stem = input_path.stem
    output_path = directory / f"{stem}.reg"

    if force or not output_path.exists():
        return output_path

    i = 1
    while True:
        versioned = directory / f"{stem}_v{i}.reg"
        if not versioned.exists():
            return versioned
        i += 1


def main() -> int:
    args = parse_args()

    feedme_path = Path(args.feedme)

    if not feedme_path.exists():
        print(f"[ERROR] File not found: {feedme_path}")
        return 1

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"[ERROR] Config not found: {args.config}")
        return 1

    color_map = get_color_map(config)

    with open(feedme_path) as f:
        lines = f.readlines()

    components = parse_components(lines)

    if not components:
        print("[WARN] No components found in feedme file")
        return 1

    regions: list[str] = []
    for comp in components:
        region_str = component_to_region(comp, color_map)
        if region_str:
            regions.append(region_str)

    output_path = resolve_output_path(feedme_path, args.output, args.force)

    with open(output_path, "w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" ')
        f.write('select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        f.write("image\n")
        for region in regions:
            f.write(region + "\n")

    print(f"Written: {output_path} ({len(regions)} regions)")
    return 0


if __name__ == "__main__":
    exit(main())
