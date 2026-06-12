# tests/test_component_builders.py

import numpy as np
import astropy.units as u
from regions import PixCoord, EllipsePixelRegion, PointPixelRegion

from galfit_builder.builders.sersic_builder import sersic_from_region
from galfit_builder.builders.psf_builder import psf_from_region


def test_sersic_from_region():
    image = np.ones((100, 100))

    region = EllipsePixelRegion(
        center=PixCoord(x=25, y=30),
        width=10,
        height=4,
        angle=30 * u.deg,
    )

    config = {
        "photometry": {
            "flux_scale": 2.0,
            "fallback_mag": 30.0,
        },
        "galfit": {
            "zeropoint": 25.0,
        },
        "defaults": {
            "sersic": {
                "sersic_index": 2,
                "include_in_output": True,
            }
        },
        "overrides": {
            "sersic": {}
        },
    }

    component = sersic_from_region(
        region,
        component_number=1,
        image_data=image,
        config=config,
    )

    assert component.component_number == 1
    assert component.x_pos.value == 26
    assert component.y_pos.value == 31
    assert component.axis_ratio.value == 0.4
    assert component.pos_angle.value == 300
    assert component.sersic_index.value == 2
    assert component.include_in_output is True


def test_psf_from_region():
    region = PointPixelRegion(
        center=PixCoord(x=10, y=20),
    )

    component = psf_from_region(
        region,
        component_number=2,
        overrides=None,
    )

    assert component.component_number == 2
    assert component.x_pos.value == 11
    assert component.y_pos.value == 21
    assert component.mag.value == 27
    assert component.include_in_output is True
