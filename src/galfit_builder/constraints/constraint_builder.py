"""Build GALFIT constraint files from config and components."""

from __future__ import annotations
from pathlib import Path
from typing import Any

from galfit_builder.components.base import GalfitComponent


PARAM_MAP = {
    "sersic": {
        "x": "x", "y": "y", "mag": "mag", "re": "re", "n": "n",
        "q": "q", "pa": "pa", "re_scale": "re"
    },
    "psf": {
        "x": "x", "y": "y", "mag": "mag"
    },
    "devauc": {
        "x": "x", "y": "y", "mag": "mag", "re": "re",
        "q": "q", "pa": "pa", "re_scale": "re"
    },
    "expdisk": {
        "x": "x", "y": "y", "mag": "mag", "rs": "rs",
        "q": "q", "pa": "pa", "rs_scale": "rs"
    },
    "gaussian": {
        "x": "x", "y": "y", "mag": "mag", "fwhm": "fwhm",
        "q": "q", "pa": "pa"
    },
    "moffat": {
        "x": "x", "y": "y", "mag": "mag", "fwhm": "fwhm", "n": "n",
        "q": "q", "pa": "pa"
    },
    "king": {
        "x": "x", "y": "y", "mu": "mu", "rc": "rc", "rt": "rt", "alpha": "alpha",
        "q": "q", "pa": "pa", "rc_scale": "rc", "rt_scale": "rt"
    },
    "nuker": {
        "x": "x", "y": "y", "mu": "mu", "rb": "rb",
        "alpha": "alpha", "beta": "beta", "gamma": "gamma",
        "q": "q", "pa": "pa", "rb_scale": "rb"
    },
    "ferrer": {
        "x": "x", "y": "y", "mu": "mu", "rout": "rout",
        "alpha": "alpha", "beta": "beta",
        "q": "q", "pa": "pa", "rout_scale": "rout"
    },
    "edgedisk": {
        "x": "x", "y": "y", "mu": "mu", "hs": "hs", "rs": "rs",
        "pa": "pa", "hs_scale": "hs", "rs_scale": "rs"
    },
    "sky": {
        "sky": "sky", "dsky_dx": "dsky_dx", "dsky_dy": "dsky_dy"
    },
}


def get_component_type(component: GalfitComponent) -> str:
    """Get the GALFIT component type name from a component object."""
    class_name = component.__class__.__name__.lower()
    return class_name


def get_scale_value(component: GalfitComponent, param: str) -> float | None:
    """Get the initial value of a parameter for scale-based constraints."""
    param_attr_map = {
        "re": "eff_rad",
        "rs": "rs",
        "rc": "rc",
        "rt": "rt",
        "rb": "rb",
        "rout": "outer_trunc_rad",
        "hs": "disk_scale_height",
    }

    attr_name = param_attr_map.get(param)
    if attr_name and hasattr(component, attr_name):
        param_obj = getattr(component, attr_name)
        if hasattr(param_obj, 'value'):
            return param_obj.value
    return None


def format_constraint(
    component_number: int,
    param: str,
    constraint_value: str | float,
    component: GalfitComponent | None = None
) -> str | None:
    """Format a single constraint line.

    Args:
        component_number: GALFIT component number
        param: Parameter name (x, y, mag, re, n, q, pa, etc.)
        constraint_value: Constraint specification (e.g., "0.5 to 8", "-5 5", or scale factor)
        component: Component object (needed for scale-based constraints)

    Returns:
        Formatted constraint line or None if constraint can't be applied
    """
    if constraint_value is None:
        return None

    # Handle scale-based constraints (e.g., re_scale = 0.1)
    if param.endswith("_scale"):
        base_param = param.replace("_scale", "")
        if component is None:
            return None

        initial_value = get_scale_value(component, base_param)
        if initial_value is None or initial_value <= 0:
            return None

        scale = float(constraint_value)
        min_val = initial_value * scale
        max_val = initial_value / scale
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        return f"{component_number}\t{base_param}\t{min_val:.4f} to {max_val:.4f}"

    # Handle string constraints
    if isinstance(constraint_value, str):
        return f"{component_number}\t{param}\t{constraint_value}"

    return None


def build_component_constraints(
    component: GalfitComponent,
    constraints_config: dict[str, Any]
) -> list[str]:
    """Build constraint lines for a single component."""
    lines = []

    comp_type = get_component_type(component)
    comp_constraints = constraints_config.get(comp_type, {})
    param_map = PARAM_MAP.get(comp_type, {})

    for config_key, constraint_value in comp_constraints.items():
        if constraint_value is None:
            continue

        galfit_param = param_map.get(config_key)
        if galfit_param is None:
            continue

        line = format_constraint(
            component.component_number,
            config_key,
            constraint_value,
            component
        )

        if line:
            lines.append(line)

    return lines


def build_constraints(
    components: list[GalfitComponent],
    config: dict[str, Any],
    constraint_file: Path | str
) -> bool:
    """Build constraint file from components and config.

    Args:
        components: List of GalfitComponent objects
        config: Configuration dictionary with [constraints.*] sections
        constraint_file: Path to write constraint file

    Returns:
        True if constraints were written, False if no constraints to write
    """
    constraints_config = config.get("constraints", {})

    if not constraints_config:
        return False

    all_lines = []

    for component in components:
        lines = build_component_constraints(component, constraints_config)
        all_lines.extend(lines)

    if not all_lines:
        return False

    with open(constraint_file, "w") as f:
        f.write("# GALFIT constraint file\n")
        f.write("# Generated by galfit-builder\n")
        f.write("#\n")
        f.write("# Component    parameter    constraint\n")
        f.write("# " + "-" * 50 + "\n")
        for line in all_lines:
            f.write(line + "\n")

    return True
