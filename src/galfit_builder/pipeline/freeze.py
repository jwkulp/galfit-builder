from __future__ import annotations
import argparse
import re
from pathlib import Path

from astropy.io import fits
from regions import Regions, PolygonPixelRegion, PixCoord

from galfit_builder.builders.component_builder import build_components
from galfit_builder.config.loader import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze GALFIT components outside polygon regions"
    )
    parser.add_argument("feedme", help="GALFIT feedme file to freeze")
    parser.add_argument("regions", help="DS9 region file with polygons defining fit area")
    parser.add_argument("--image", "-i", help="Input image (required if adding new components)")
    parser.add_argument("--config", "-c", help="Config file (required if adding new components)")
    parser.add_argument("--constraint", help="Constraint file to update")
    parser.add_argument("--output", "-o", help="Output feedme file (default: frozen_<input>)")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def parse_feedme_components(lines: list[str]) -> list[dict]:
    """Extract component info from feedme lines."""
    components = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("# Object number:") or line.startswith("# Component number:"):
            comp_num = int(line.split(":")[1].strip())

            # Check if it's sky (next line has "0) sky")
            if i + 1 < len(lines) and "sky" in lines[i + 1].lower():
                i += 1
                continue

            # Find position line (starts with " 1)")
            start_idx = i
            pos_line_idx = None
            for j in range(i + 1, min(i + 15, len(lines))):
                if lines[j].strip().startswith("1)"):
                    pos_line_idx = j
                    break

            if pos_line_idx:
                pos_parts = lines[pos_line_idx].split(")")[1].split()
                try:
                    x, y = float(pos_parts[0]), float(pos_parts[1])
                    components.append({
                        "component_number": comp_num,
                        "start_index": start_idx,
                        "x": x,
                        "y": y,
                        "matched": False,
                    })
                except (ValueError, IndexError):
                    pass
        i += 1
    return components


def point_in_polygons(x: float, y: float, polygons: list[PolygonPixelRegion]) -> bool:
    """Check if point is inside any polygon."""
    coord = PixCoord(x, y)
    return any(poly.contains(coord) for poly in polygons)


def freeze_line(line: str, freeze: bool) -> str:
    """Set fit toggle(s) in a parameter line to 0 (freeze) or 1 (free).

    Handles both single toggle "value toggle" and dual toggle "x y tx ty" formats.
    """
    # Skip comment lines and empty lines
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return line

    # Match parameter lines like " 1) value toggle" or " 3) value toggle"
    # or position lines like " 1) x y tx ty"
    match = re.match(r'^(\s*\d+\))\s+(.+)$', line)
    if not match:
        return line

    prefix = match.group(1)
    rest = match.group(2)

    # Split into value(s) and comment
    if "#" in rest:
        params_part, comment = rest.split("#", 1)
        comment = "# " + comment.strip()
    else:
        params_part = rest
        comment = ""

    parts = params_part.split()
    if len(parts) < 2:
        return line

    toggle_val = "0" if freeze else "1"

    # Position line: x y tx ty
    if len(parts) >= 4 and parts[2] in ("0", "1") and parts[3] in ("0", "1"):
        new_parts = [parts[0], parts[1], toggle_val, toggle_val]
        new_line = f"{prefix} {new_parts[0]}  {new_parts[1]}  {new_parts[2]} {new_parts[3]}  {comment}"
    # Single value line: value toggle
    elif len(parts) >= 2 and parts[-1] in ("0", "1", "-1"):
        # Keep -1 (fixed) as is
        if parts[-1] == "-1":
            return line
        new_parts = parts[:-1] + [toggle_val]
        new_line = f"{prefix} {'   '.join(new_parts)}  {comment}"
    else:
        return line

    return new_line.rstrip() + "\n"


def freeze_component(lines: list[str], start_idx: int, freeze: bool) -> list[str]:
    """Freeze or unfreeze all parameters of a component."""
    result = lines.copy()

    # Find the component block (until next component or end)
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith("# Object number:") or line.startswith("# Component number:"):
            end_idx = i
            break

    # Process parameter lines in the component
    for i in range(start_idx, end_idx):
        if lines[i].strip().startswith(("1)", "2)", "3)", "4)", "5)", "6)", "7)", "8)", "9)", "10)")):
            result[i] = freeze_line(lines[i], freeze)

    return result


def apply_polygon_freezing(
    lines: list[str],
    components: list[dict],
    polygons: list[PolygonPixelRegion]
) -> list[str]:
    """Freeze components outside polygons, free those inside."""
    result = lines.copy()

    for comp in components:
        inside = point_in_polygons(comp["x"], comp["y"], polygons)
        result = freeze_component(result, comp["start_index"], freeze=not inside)

    return result


def match_with_regions(
    feedme_comps: list[dict],
    region_comps: list,
    tolerance: float = 1.0
) -> list[bool]:
    """Match feedme components with region components by position.

    Returns list of booleans indicating which region components are matched.
    """
    matched = [False] * len(region_comps)

    for i, reg in enumerate(region_comps):
        reg_x = reg.center.x + 1  # Convert to GALFIT coords
        reg_y = reg.center.y + 1

        for comp in feedme_comps:
            dx = abs(comp["x"] - reg_x)
            dy = abs(comp["y"] - reg_y)
            if dx < tolerance and dy < tolerance:
                matched[i] = True
                comp["matched"] = True
                break

    return matched


def reindex_components(lines: list[str], deleted_num: int) -> list[str]:
    """Decrement component numbers greater than deleted_num."""
    result = []
    for line in lines:
        if line.strip().startswith("# Object number:") or line.strip().startswith("# Component number:"):
            match = re.search(r':\s*(\d+)', line)
            if match:
                num = int(match.group(1))
                if num > deleted_num:
                    line = re.sub(r':\s*\d+', f': {num - 1}', line)
        result.append(line)
    return result


def delete_component(lines: list[str], start_idx: int, comp_num: int) -> list[str]:
    """Delete a component block from feedme lines."""
    # Find end of component block
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith("# Object number:") or line.startswith("# Component number:"):
            end_idx = i
            break

    # Remove the block
    result = lines[:start_idx] + lines[end_idx:]

    # Reindex remaining components
    return reindex_components(result, comp_num)


def reindex_constraints(lines: list[str], deleted_num: int) -> list[str]:
    """Decrement component numbers in constraint file."""
    result = []
    for line in lines:
        parts = line.strip().split()
        if parts and parts[0].isdigit():
            num = int(parts[0])
            if num > deleted_num:
                parts[0] = str(num - 1)
                line = "\t".join(parts) + "\n"
            elif num == deleted_num:
                continue  # Skip deleted component's constraints
        result.append(line)
    return result


def resolve_output_path(input_path: Path, output: str | None, force: bool) -> Path:
    """Determine output path, handling versioning."""
    if output:
        return Path(output)

    directory = input_path.parent
    stem = f"frozen_{input_path.stem}"
    suffix = input_path.suffix

    base = directory / f"{stem}{suffix}"

    if force or not base.exists():
        return base

    i = 1
    while True:
        path = directory / f"{stem}_v{i}{suffix}"
        if not path.exists():
            return path
        i += 1


def main():
    args = parse_args()

    feedme_path = Path(args.feedme)
    regions_path = Path(args.regions)

    # Read feedme
    with open(feedme_path) as f:
        feedme_lines = f.readlines()

    # Read regions
    all_regions = Regions.read(regions_path, format="ds9")

    # Green polygons define freeze zones
    polygons = [
        r for r in all_regions
        if isinstance(r, PolygonPixelRegion)
        and (r.visual.get("edgecolor") or r.visual.get("facecolor") or "").lower() == "green"
    ]

    # Non-polygon regions are components
    component_regions = [r for r in all_regions if not isinstance(r, PolygonPixelRegion)]

    # Parse existing components
    feedme_comps = parse_feedme_components(feedme_lines)

    # Match with regions
    region_matched: list[bool] = []
    unmatched: list[dict] = []

    if component_regions:
        region_matched = match_with_regions(feedme_comps, component_regions)

        # Delete unmatched feedme components (in reverse order to preserve indices)
        unmatched = [c for c in feedme_comps if not c["matched"]]
        for comp in sorted(unmatched, key=lambda c: c["start_index"], reverse=True):
            feedme_lines = delete_component(
                feedme_lines, comp["start_index"], comp["component_number"]
            )

        # Re-parse after deletions
        feedme_comps = parse_feedme_components(feedme_lines)

    # Apply polygon freezing
    if polygons:
        feedme_lines = apply_polygon_freezing(feedme_lines, feedme_comps, polygons)

    # Add new components from unmatched regions (if image and config provided)
    if args.image and args.config and component_regions and region_matched:
        new_regions = [r for i, r in enumerate(component_regions) if not region_matched[i]]
        if new_regions:
            config = load_config(args.config)
            image_data = fits.getdata(args.image)

            # Get last component number
            last_num = max((c["component_number"] for c in feedme_comps), default=1)

            # Build new components
            new_comps = build_components(new_regions, image_data, config)

            # Renumber and append
            for i, comp in enumerate(new_comps):
                comp.component_number = last_num + i + 1
                feedme_lines.append(str(comp) + "\n")

    # Write output
    output_path = resolve_output_path(feedme_path, args.output, args.force)
    with open(output_path, "w") as f:
        f.writelines(feedme_lines)

    print(f"Written: {output_path}")

    # Handle constraint file if provided
    if args.constraint:
        constraint_path = Path(args.constraint)
        with open(constraint_path) as f:
            constraint_lines = f.readlines()

        # Reindex for deleted components
        for comp in sorted(unmatched, key=lambda c: c["component_number"], reverse=True):
            constraint_lines = reindex_constraints(constraint_lines, comp["component_number"])

        constraint_output = resolve_output_path(constraint_path, None, args.force)
        with open(constraint_output, "w") as f:
            f.writelines(constraint_lines)

        print(f"Written: {constraint_output}")


if __name__ == "__main__":
    main()
