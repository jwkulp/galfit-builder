"""I/O utilities for reading regions and writing feedme files."""


def __getattr__(name: str):
    """Lazy import to avoid requiring regions at import time."""
    if name == "read_regions":
        from galfit_builder.io.parse_regions import read_regions
        return read_regions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["read_regions"]
