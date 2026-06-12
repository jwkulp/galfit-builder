"""Component builders for converting regions to GALFIT components."""


def __getattr__(name: str):
    """Lazy import to avoid requiring regions at import time."""
    if name in ("build_components", "infer_component_type", "build_single_component"):
        from galfit_builder.builders.component_builder import (
            build_components,
            infer_component_type,
            build_single_component,
        )
        if name == "build_components":
            return build_components
        if name == "build_single_component":
            return build_single_component
        return infer_component_type

    if name == "build_sky":
        from galfit_builder.builders.build_sky import build_sky
        return build_sky

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_components",
    "build_single_component",
    "build_sky",
    "infer_component_type",
]
