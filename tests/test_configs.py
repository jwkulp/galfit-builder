"""Tests with different config variations and edge cases."""

import tempfile
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.io import fits
from regions import PixCoord, EllipsePixelRegion, PointPixelRegion

from galfit_builder.components.base import GalfitParam
from galfit_builder.components.sersic import Sersic
from galfit_builder.builders.build_sersic import build_sersic
from galfit_builder.builders.build_psf import build_psf
from galfit_builder.builders.build_sky import build_sky
from galfit_builder.builders.build_king import build_king
from galfit_builder.builders.build_ferrer import build_ferrer
from galfit_builder.builders.component_builder import build_components
from galfit_builder.controls.controls import GalfitControls
from galfit_builder.constraints.constraint_builder import build_constraints


def make_base_config():
    """Create a minimal valid config dict."""
    return {
        "galfit_input_controls": {
            "zeropoint": 25.0,
            "plate_scale": [0.03, 0.03],
        },
        "photometry": {
            "fallback_mag": 30.0,
            "fallback_mu": 25.0,
        },
        "region_colors": {
            "cyan": "sersic",
            "magenta": "king",
            "blue": "ferrer",
        },
        "defaults": {
            "sersic": {
                "sersic_index": 2.5,
                "x_pos_freeze": False,
                "y_pos_freeze": False,
                "mag_freeze": False,
                "eff_rad_freeze": False,
                "sersic_index_freeze": False,
                "axis_ratio_freeze": False,
                "pos_angle_freeze": False,
                "include_in_output": True,
            },
            "psf": {
                "mag": 25.0,
                "x_pos_freeze": False,
                "y_pos_freeze": False,
                "mag_freeze": False,
                "include_in_output": True,
            },
            "sky": {
                "include": True,
                "background": None,
                "dsky_dx": 0.0,
                "dsky_dy": 0.0,
                "background_freeze": False,
                "dsky_dx_freeze": True,
                "dsky_dy_freeze": True,
                "include_in_output": True,
            },
            "king": {
                "rc_ratio": 0.1,
                "rt_ratio": 1.0,
                "min_rc": 1.0,
                "min_rt": 3.0,
                "alpha": 2.0,
                "x_pos_freeze": False,
                "y_pos_freeze": False,
                "mu_freeze": False,
                "rc_freeze": False,
                "rt_freeze": False,
                "alpha_freeze": False,
                "axis_ratio_freeze": False,
                "pos_angle_freeze": False,
                "include_in_output": True,
            },
            "ferrer": {
                "alpha": 2.0,
                "beta": 2.0,
                "x_pos_freeze": False,
                "y_pos_freeze": False,
                "cen_surf_bright_freeze": False,
                "outer_trunc_rad_freeze": False,
                "alpha_freeze": True,
                "beta_freeze": True,
                "axis_ratio_freeze": False,
                "pos_angle_freeze": False,
                "include_in_output": True,
            },
        },
        "overrides": {},
    }


def make_test_image(shape=(100, 100), background=0.001, sources=None):
    """Create a test image with optional sources."""
    image = np.ones(shape) * background

    if sources:
        for src in sources:
            x, y, flux = src["x"], src["y"], src["flux"]
            size = src.get("size", 3)
            yy, xx = np.ogrid[:shape[0], :shape[1]]
            r2 = (xx - x)**2 + (yy - y)**2
            image += flux * np.exp(-r2 / (2 * size**2))

    return image


# =============================================================================
# Test Different Zeropoints
# =============================================================================

class TestZeropoints:
    def test_high_zeropoint(self):
        config_high = make_base_config()
        config_high["galfit_input_controls"]["zeropoint"] = 30.0

        config_low = make_base_config()
        config_low["galfit_input_controls"]["zeropoint"] = 25.0

        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20, height=16, angle=0 * u.deg,
        )

        sersic_high = build_sersic(1, region, config_high, image)
        sersic_low = build_sersic(1, region, config_low, image)

        # Higher zeropoint = fainter magnitudes (higher number)
        # mag = -2.5*log10(flux) + zeropoint
        assert sersic_high.mag.value > sersic_low.mag.value

    def test_low_zeropoint(self):
        config = make_base_config()
        config["galfit_input_controls"]["zeropoint"] = 20.0

        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20, height=16, angle=0 * u.deg,
        )

        sersic = build_sersic(1, region, config, image)

        # With zp=20 and reasonable flux, should get a brighter mag
        assert sersic.mag.value < 20


# =============================================================================
# Test Different Plate Scales
# =============================================================================

class TestPlateScales:
    def test_different_plate_scales(self):
        config1 = make_base_config()
        config1["galfit_input_controls"]["plate_scale"] = [0.03, 0.03]

        config2 = make_base_config()
        config2["galfit_input_controls"]["plate_scale"] = [0.1, 0.1]

        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20, height=18, angle=0 * u.deg,
        )

        # Build king with different plate scales
        king1 = build_king(1, region, config1, image)
        king2 = build_king(1, region, config2, image)

        # Surface brightness depends on plate scale
        # Larger plate scale = smaller area per pixel = brighter per arcsec^2
        assert king1.mu.value != king2.mu.value


# =============================================================================
# Test Override Configs
# =============================================================================

class TestOverrides:
    def test_sersic_index_override(self):
        config = make_base_config()
        config["defaults"]["sersic"]["sersic_index"] = 2.5
        config["overrides"]["sersic"] = {"sersic_index": 4.0}

        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20, height=16, angle=0 * u.deg,
        )

        sersic = build_sersic(1, region, config, image)

        assert sersic.sersic_index.value == 4.0

    def test_psf_mag_override(self):
        config = make_base_config()
        config["defaults"]["psf"]["mag"] = 25.0
        config["overrides"]["psf"] = {"mag": 22.0}

        region = PointPixelRegion(center=PixCoord(x=49, y=49))

        psf = build_psf(1, region, config)

        assert psf.mag.value == 22.0

    def test_freeze_override(self):
        config = make_base_config()
        config["defaults"]["sersic"]["x_pos_freeze"] = False
        config["overrides"]["sersic"] = {"x_pos_freeze": True}

        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20, height=16, angle=0 * u.deg,
        )

        sersic = build_sersic(1, region, config, image)

        assert sersic.x_pos.freeze is True


# =============================================================================
# Test Ferrer Specific Params
# =============================================================================

class TestFerrerConfig:
    def test_alpha_beta_defaults(self):
        config = make_base_config()
        config["defaults"]["ferrer"]["alpha"] = 3.0
        config["defaults"]["ferrer"]["beta"] = 1.5

        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20, height=18, angle=0 * u.deg,
        )

        ferrer = build_ferrer(1, region, config, image)

        assert ferrer.alpha.value == 3.0
        assert ferrer.beta.value == 1.5

    def test_ferrer_freeze_alpha_beta(self):
        config = make_base_config()
        config["defaults"]["ferrer"]["alpha_freeze"] = True
        config["defaults"]["ferrer"]["beta_freeze"] = True

        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20, height=18, angle=0 * u.deg,
        )

        ferrer = build_ferrer(1, region, config, image)

        assert ferrer.alpha.freeze is True
        assert ferrer.beta.freeze is True


# =============================================================================
# Test King Specific Params
# =============================================================================

class TestKingConfig:
    def test_rc_rt_ratios(self):
        config = make_base_config()
        config["defaults"]["king"]["rc_ratio"] = 0.2
        config["defaults"]["king"]["rt_ratio"] = 1.5

        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20, height=18, angle=0 * u.deg,  # a = 10
        )

        king = build_king(1, region, config, image)

        # rc = max(a * rc_ratio, min_rc) = max(10 * 0.2, 1.0) = 2.0
        assert king.rc.value == 2.0
        # rt = max(a * rt_ratio, min_rt) = max(10 * 1.5, 3.0) = 15.0
        assert king.rt.value == 15.0

    def test_king_min_values(self):
        config = make_base_config()
        config["defaults"]["king"]["rc_ratio"] = 0.01  # Very small
        config["defaults"]["king"]["min_rc"] = 1.0
        config["defaults"]["king"]["rt_ratio"] = 0.1  # Very small
        config["defaults"]["king"]["min_rt"] = 3.0

        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=10, height=8, angle=0 * u.deg,  # a = 5
        )

        king = build_king(1, region, config, image)

        # Should use min values
        assert king.rc.value >= 1.0
        assert king.rt.value >= 3.0


# =============================================================================
# Test Sky Configuration
# =============================================================================

class TestSkyConfig:
    def test_sky_disabled(self):
        config = make_base_config()
        config["defaults"]["sky"]["include"] = False

        image = make_test_image()
        region = PointPixelRegion(center=PixCoord(x=49, y=49))

        components = build_components([region], image, config)

        # Should only have PSF, no sky
        assert len(components) == 1
        assert components[0].__class__.__name__ == "PSF"

    def test_sky_fixed_background(self):
        config = make_base_config()
        config["defaults"]["sky"]["background"] = 0.005

        image = make_test_image(background=0.001)

        sky = build_sky(1, None, config, image)

        # Should use config value, not estimated
        assert sky.background.value == 0.005

    def test_sky_gradient_freeze(self):
        config = make_base_config()
        config["defaults"]["sky"]["dsky_dx_freeze"] = True
        config["defaults"]["sky"]["dsky_dy_freeze"] = True

        image = make_test_image()

        sky = build_sky(1, None, config, image)

        assert sky.dsky_dx.freeze is True
        assert sky.dsky_dy.freeze is True


# =============================================================================
# Test Component Colors
# =============================================================================

class TestRegionColors:
    def test_color_mapping(self):
        config = make_base_config()
        config["region_colors"] = {
            "cyan": "sersic",
            "magenta": "king",
            "blue": "ferrer",
            "red": "nuker",
        }

        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])

        # Cyan region -> sersic
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20, height=16, angle=0 * u.deg,
        )
        region.visual["edgecolor"] = "cyan"

        components = build_components([region], image, config)

        # Should have sky + sersic
        assert len(components) == 2
        assert components[1].__class__.__name__ == "Sersic"


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_zero_flux_region(self):
        config = make_base_config()
        config["photometry"]["fallback_mag"] = 30.0

        # Image with zero flux
        image = np.zeros((100, 100))

        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20, height=16, angle=0 * u.deg,
        )

        sersic = build_sersic(1, region, config, image)

        # Should use fallback magnitude
        assert sersic.mag.value == 30.0

    def test_small_region(self):
        config = make_base_config()

        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])

        # Very small region
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=2, height=2, angle=0 * u.deg,
        )

        sersic = build_sersic(1, region, config, image)

        assert sersic.eff_rad.value == 1.0  # a = width/2 = 1

    def test_large_region(self):
        config = make_base_config()

        image = make_test_image(shape=(200, 200), sources=[{"x": 100, "y": 100, "flux": 1000, "size": 20}])

        region = EllipsePixelRegion(
            center=PixCoord(x=99, y=99),
            width=100, height=80, angle=0 * u.deg,
        )

        sersic = build_sersic(1, region, config, image)

        assert sersic.eff_rad.value == 50.0  # a = width/2 = 50

    def test_rotated_region(self):
        config = make_base_config()

        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])

        # Region rotated by 45 degrees
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20, height=10, angle=45 * u.deg,
        )

        sersic = build_sersic(1, region, config, image)

        # Position angle should be adjusted
        # DS9 angle -> GALFIT angle: (angle - 90) % 360
        expected_pa = (45 - 90) % 360
        assert abs(sersic.pos_angle.value - expected_pa) < 0.1

    def test_multiple_components(self):
        config = make_base_config()
        config["defaults"]["sky"]["include"] = True

        image = make_test_image(
            sources=[
                {"x": 30, "y": 30, "flux": 100},
                {"x": 70, "y": 70, "flux": 50},
            ]
        )

        regions = [
            PointPixelRegion(center=PixCoord(x=29, y=29)),  # PSF
            EllipsePixelRegion(
                center=PixCoord(x=69, y=69),
                width=20, height=16, angle=0 * u.deg,
            ),  # Sersic (default for no color)
        ]

        components = build_components(regions, image, config)

        # Should have: sky, psf, sersic
        assert len(components) == 3
        assert components[0].__class__.__name__ == "Sky"
        assert components[1].__class__.__name__ == "PSF"
        assert components[2].__class__.__name__ == "Sersic"

        # Check component numbers
        assert components[0].component_number == 1
        assert components[1].component_number == 2
        assert components[2].component_number == 3


# =============================================================================
# Test Controls with Different Configs
# =============================================================================

class TestControlsConfigs:
    def test_controls_with_fit_region(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_fits = tmpdir / "input.fits"
            hdu = fits.PrimaryHDU(np.zeros((100, 100)))
            hdu.writeto(input_fits)

            controls = GalfitControls(
                working_dir=tmpdir,
                input_data_image=input_fits,
                output_data_image=tmpdir / "output.fits",
                zeropoint=25.0,
                plate_scale=(0.03, 0.03),
                fit_region=(10, 90, 10, 90),
            )

            controls.validate_inputs()

            output = str(controls)
            assert "10 90 10 90" in output

    def test_controls_with_conv_box(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_fits = tmpdir / "input.fits"
            hdu = fits.PrimaryHDU(np.zeros((100, 100)))
            hdu.writeto(input_fits)

            controls = GalfitControls(
                working_dir=tmpdir,
                input_data_image=input_fits,
                output_data_image=tmpdir / "output.fits",
                zeropoint=25.0,
                plate_scale=(0.03, 0.03),
                conv_box_size=(50, 50),
            )

            controls.validate_inputs()

            output = str(controls)
            assert "50 50" in output

    def test_controls_display_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_fits = tmpdir / "input.fits"
            hdu = fits.PrimaryHDU(np.zeros((100, 100)))
            hdu.writeto(input_fits)

            for display_type in ["regular", "curses", "both"]:
                controls = GalfitControls(
                    working_dir=tmpdir,
                    input_data_image=input_fits,
                    output_data_image=tmpdir / "output.fits",
                    zeropoint=25.0,
                    plate_scale=(0.03, 0.03),
                    display_type=display_type,
                )

                controls.validate_inputs()

                output = str(controls)
                assert display_type in output


# =============================================================================
# Test Constraints with Different Configs
# =============================================================================

class TestConstraintsConfigs:
    def test_soft_constraints(self):
        config = make_base_config()
        config["constraints"] = {
            "sersic": {
                "n": "0.5 to 8",
                "q": "0.1 to 1.0",
                "re": "-5 5",  # Relative constraint
            }
        }

        sersic = Sersic(
            component_number=2,
            x_pos=GalfitParam(50.0),
            y_pos=GalfitParam(50.0),
            mag=GalfitParam(20.0),
            eff_rad=GalfitParam(10.0),
            sersic_index=GalfitParam(2.5),
            axis_ratio=GalfitParam(0.8),
            pos_angle=GalfitParam(45.0),
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            constraint_file = Path(f.name)

        try:
            result = build_constraints([sersic], config, constraint_file)

            assert result is True
            content = constraint_file.read_text()
            assert "0.5 to 8" in content
            assert "0.1 to 1.0" in content
        finally:
            constraint_file.unlink(missing_ok=True)

    def test_scale_constraints(self):
        config = make_base_config()
        config["constraints"] = {
            "sersic": {
                "re_scale": 0.2,  # Scale-based constraint
            }
        }

        sersic = Sersic(
            component_number=2,
            x_pos=GalfitParam(50.0),
            y_pos=GalfitParam(50.0),
            mag=GalfitParam(20.0),
            eff_rad=GalfitParam(10.0),  # Initial value = 10
            sersic_index=GalfitParam(2.5),
            axis_ratio=GalfitParam(0.8),
            pos_angle=GalfitParam(45.0),
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            constraint_file = Path(f.name)

        try:
            result = build_constraints([sersic], config, constraint_file)

            assert result is True
            content = constraint_file.read_text()
            # re_scale = 0.2 with initial = 10 -> min = 2.0, max = 50.0
            assert "2.0000 to 50.0000" in content
        finally:
            constraint_file.unlink(missing_ok=True)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
