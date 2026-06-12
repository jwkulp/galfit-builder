"""Comprehensive tests for galfit-builder modules."""

import tempfile
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.io import fits
from regions import PixCoord, EllipsePixelRegion, PointPixelRegion, PolygonPixelRegion

from galfit_builder.components.base import GalfitParam, GalfitComponent
from galfit_builder.components.sersic import Sersic
from galfit_builder.components.psf import PSF
from galfit_builder.components.sky import Sky
from galfit_builder.components.king import King
from galfit_builder.components.ferrer import Ferrer
from galfit_builder.components.nuker import Nuker
from galfit_builder.components.devauc import Devauc
from galfit_builder.components.expdisk import Expdisk
from galfit_builder.components.gaussian import Gaussian
from galfit_builder.components.moffat import Moffat

from galfit_builder.builders.build_sersic import build_sersic
from galfit_builder.builders.build_psf import build_psf
from galfit_builder.builders.build_sky import build_sky, estimate_sky_background
from galfit_builder.builders.build_king import build_king
from galfit_builder.builders.build_ferrer import build_ferrer
from galfit_builder.builders.build_nuker import build_nuker
from galfit_builder.builders.build_devauc import build_devauc
from galfit_builder.builders.build_expdisk import build_expdisk
from galfit_builder.builders.build_gaussian import build_gaussian
from galfit_builder.builders.build_moffat import build_moffat
from galfit_builder.builders.component_builder import build_components, infer_component_type
from galfit_builder.builders.region_utils import ellipse_geometry, estimate_ellipse_flux, flux_to_mag
from galfit_builder.builders.builder_utils import ParamResolver

from galfit_builder.controls.controls import GalfitControls
from galfit_builder.constraints.constraint_builder import build_constraints


# =============================================================================
# Test Config Templates
# =============================================================================

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
            "red": "nuker",
            "yellow": "devauc",
            "green": "expdisk",
            "orange": "gaussian",
            "purple": "moffat",
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
            "nuker": {
                "alpha": 2.0,
                "beta": 1.5,
                "gamma": 0.5,
                "rb_ratio": 0.3,
                "min_rb": 1.0,
                "x_pos_freeze": False,
                "y_pos_freeze": False,
                "mu_freeze": False,
                "rb_freeze": False,
                "alpha_freeze": False,
                "beta_freeze": False,
                "gamma_freeze": False,
                "axis_ratio_freeze": False,
                "pos_angle_freeze": False,
                "include_in_output": True,
            },
            "devauc": {
                "x_pos_freeze": False,
                "y_pos_freeze": False,
                "mag_freeze": False,
                "eff_rad_freeze": False,
                "axis_ratio_freeze": False,
                "pos_angle_freeze": False,
                "include_in_output": True,
            },
            "expdisk": {
                "x_pos_freeze": False,
                "y_pos_freeze": False,
                "mag_freeze": False,
                "rs_freeze": False,
                "axis_ratio_freeze": False,
                "pos_angle_freeze": False,
                "include_in_output": True,
            },
            "gaussian": {
                "x_pos_freeze": False,
                "y_pos_freeze": False,
                "mag_freeze": False,
                "fwhm_freeze": False,
                "axis_ratio_freeze": False,
                "pos_angle_freeze": False,
                "include_in_output": True,
            },
            "moffat": {
                "powerlaw": 2.5,
                "x_pos_freeze": False,
                "y_pos_freeze": False,
                "mag_freeze": False,
                "fwhm_freeze": False,
                "powerlaw_freeze": False,
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
# Test GalfitParam
# =============================================================================

class TestGalfitParam:
    def test_basic_creation(self):
        p = GalfitParam(10.5)
        assert p.value == 10.5
        assert p.freeze is False

    def test_frozen_param(self):
        p = GalfitParam(20.0, freeze=True)
        assert p.freeze is True

    def test_integer_conversion(self):
        p = GalfitParam(10)
        assert isinstance(p.value, float)
        assert p.value == 10.0

    def test_invalid_value_raises(self):
        import pytest
        with pytest.raises(TypeError):
            GalfitParam("not a number")

    def test_infinite_value_raises(self):
        import pytest
        with pytest.raises(ValueError):
            GalfitParam(float('inf'))


# =============================================================================
# Test Components
# =============================================================================

class TestSersicComponent:
    def test_basic_sersic(self):
        s = Sersic(
            component_number=1,
            x_pos=GalfitParam(50.0),
            y_pos=GalfitParam(50.0),
            mag=GalfitParam(20.0),
            eff_rad=GalfitParam(10.0),
            sersic_index=GalfitParam(2.5),
            axis_ratio=GalfitParam(0.8),
            pos_angle=GalfitParam(45.0),
        )
        output = s.to_galfit()
        assert "sersic" in output
        assert "50.00" in output
        assert "20.0000" in output

    def test_frozen_params(self):
        s = Sersic(
            component_number=2,
            x_pos=GalfitParam(50.0, freeze=True),
            y_pos=GalfitParam(50.0, freeze=True),
            mag=GalfitParam(20.0),
            eff_rad=GalfitParam(10.0),
            sersic_index=GalfitParam(2.5),
            axis_ratio=GalfitParam(0.8),
            pos_angle=GalfitParam(45.0),
        )
        output = s.to_galfit()
        lines = output.split('\n')
        pos_line = [l for l in lines if "position x, y" in l][0]
        assert "0 0" in pos_line


class TestPSFComponent:
    def test_basic_psf(self):
        p = PSF(
            component_number=1,
            x_pos=GalfitParam(25.0),
            y_pos=GalfitParam(30.0),
            mag=GalfitParam(22.0),
        )
        output = p.to_galfit()
        assert "psf" in output
        assert "25.00" in output


class TestSkyComponent:
    def test_basic_sky(self):
        s = Sky(
            component_number=1,
            background=GalfitParam(0.001),
            dsky_dx=GalfitParam(0.0),
            dsky_dy=GalfitParam(0.0),
        )
        output = s.to_galfit()
        assert "sky" in output


class TestKingComponent:
    def test_basic_king(self):
        k = King(
            component_number=1,
            x_pos=GalfitParam(50.0),
            y_pos=GalfitParam(50.0),
            mu=GalfitParam(20.0),
            rc=GalfitParam(2.0),
            rt=GalfitParam(10.0),
            alpha=GalfitParam(2.0),
            axis_ratio=GalfitParam(0.9),
            pos_angle=GalfitParam(0.0),
        )
        output = k.to_galfit()
        assert "king" in output


class TestFerrerComponent:
    def test_basic_ferrer(self):
        f = Ferrer(
            component_number=1,
            x_pos=GalfitParam(50.0),
            y_pos=GalfitParam(50.0),
            cen_surf_bright=GalfitParam(21.0),
            outer_trunc_rad=GalfitParam(5.0),
            alpha=GalfitParam(2.0),
            beta=GalfitParam(2.0),
            axis_ratio=GalfitParam(0.9),
            pos_angle=GalfitParam(0.0),
        )
        output = f.to_galfit()
        assert "ferrer" in output


# =============================================================================
# Test Builders
# =============================================================================

class TestSersicBuilder:
    def test_build_sersic_from_region(self):
        config = make_base_config()
        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])

        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),  # 0-indexed
            width=20,
            height=16,
            angle=30 * u.deg,
        )

        sersic = build_sersic(1, region, config, image)

        assert sersic.component_number == 1
        assert abs(sersic.x_pos.value - 50) < 0.01
        assert abs(sersic.y_pos.value - 50) < 0.01
        assert sersic.axis_ratio.value == 0.8  # 8/10


class TestPSFBuilder:
    def test_build_psf_from_region(self):
        config = make_base_config()

        region = PointPixelRegion(center=PixCoord(x=24, y=29))

        psf = build_psf(2, region, config)

        assert psf.component_number == 2
        assert psf.x_pos.value == 25
        assert psf.y_pos.value == 30
        assert psf.mag.value == 25.0


class TestSkyBuilder:
    def test_sky_estimation(self):
        image = np.ones((100, 100)) * 0.001
        image[40:60, 40:60] = 1.0  # bright center

        bg = estimate_sky_background(image)

        assert abs(bg - 0.001) < 0.01

    def test_build_sky(self):
        config = make_base_config()
        image = np.ones((100, 100)) * 0.002

        sky = build_sky(1, None, config, image)

        assert sky.component_number == 1
        assert abs(sky.background.value - 0.002) < 0.001


class TestKingBuilder:
    def test_build_king_from_region(self):
        config = make_base_config()
        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 50}])

        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20,
            height=18,
            angle=0 * u.deg,
        )

        king = build_king(1, region, config, image)

        assert king.component_number == 1
        assert abs(king.x_pos.value - 50) < 0.01


class TestFerrerBuilder:
    def test_build_ferrer_from_region(self):
        config = make_base_config()
        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 50}])

        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20,
            height=18,
            angle=0 * u.deg,
        )

        ferrer = build_ferrer(1, region, config, image)

        assert ferrer.component_number == 1
        assert ferrer.alpha.value == 2.0
        assert ferrer.beta.value == 2.0


class TestNukerBuilder:
    def test_build_nuker_from_region(self):
        config = make_base_config()
        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 50}])

        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20,
            height=18,
            angle=0 * u.deg,
        )

        nuker = build_nuker(1, region, config, image)

        assert nuker.component_number == 1
        assert nuker.alpha.value == 2.0
        assert nuker.beta.value == 1.5
        assert nuker.gamma.value == 0.5


# =============================================================================
# Test Component Builder (integration)
# =============================================================================

class TestComponentBuilder:
    def test_infer_component_type_psf(self):
        region = PointPixelRegion(center=PixCoord(x=10, y=20))
        color_map = {"cyan": "sersic"}

        assert infer_component_type(region, color_map) == "psf"

    def test_infer_component_type_ellipse_by_color(self):
        region = EllipsePixelRegion(
            center=PixCoord(x=10, y=20),
            width=10,
            height=8,
            angle=0 * u.deg,
        )
        region.visual["edgecolor"] = "cyan"
        color_map = {"cyan": "sersic", "magenta": "king"}

        assert infer_component_type(region, color_map) == "sersic"

    def test_build_components_with_sky(self):
        config = make_base_config()
        config["defaults"]["sky"]["include"] = True

        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])

        region = PointPixelRegion(center=PixCoord(x=49, y=49))

        components = build_components([region], image, config)

        assert len(components) == 2
        assert components[0].__class__.__name__ == "Sky"
        assert components[1].__class__.__name__ == "PSF"

    def test_build_components_without_sky(self):
        config = make_base_config()
        config["defaults"]["sky"]["include"] = False

        image = make_test_image()
        region = PointPixelRegion(center=PixCoord(x=49, y=49))

        components = build_components([region], image, config)

        assert len(components) == 1
        assert components[0].__class__.__name__ == "PSF"


# =============================================================================
# Test Region Utils
# =============================================================================

class TestEllipseGeometry:
    def test_basic_geometry(self):
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=20,
            height=10,
            angle=0 * u.deg,
        )

        geom = ellipse_geometry(region)

        assert geom.x_pos == 50
        assert geom.y_pos == 50
        assert geom.a == 10
        assert geom.b == 5

    def test_swapped_axes(self):
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=10,
            height=20,
            angle=0 * u.deg,
        )

        geom = ellipse_geometry(region)

        assert geom.a == 10
        assert geom.b == 5


class TestFluxEstimation:
    def test_positive_flux(self):
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=49),
            width=10,
            height=10,
            angle=0 * u.deg,
        )

        image = make_test_image(sources=[{"x": 50, "y": 50, "flux": 100}])

        flux, fallback = estimate_ellipse_flux(region, image)

        assert flux > 0
        assert fallback is False

    def test_flux_to_mag(self):
        mag = flux_to_mag(100.0, 25.0, 30.0)
        expected = -2.5 * np.log10(100.0) + 25.0
        assert abs(mag - expected) < 0.001

    def test_zero_flux_fallback(self):
        mag = flux_to_mag(0.0, 25.0, 30.0)
        assert mag == 30.0


# =============================================================================
# Test ParamResolver
# =============================================================================

class TestParamResolver:
    def test_default_value(self):
        defaults = {"alpha": 2.0, "beta": 1.0}
        overrides = {}

        resolver = ParamResolver(defaults, overrides)

        assert resolver.get("alpha") == 2.0
        assert resolver.get("beta") == 1.0

    def test_override_value(self):
        defaults = {"alpha": 2.0}
        overrides = {"alpha": 3.0}

        resolver = ParamResolver(defaults, overrides)

        assert resolver.get("alpha") == 3.0

    def test_auto_falls_back_to_default(self):
        defaults = {"alpha": 2.0}
        overrides = {"alpha": "auto"}

        resolver = ParamResolver(defaults, overrides)

        assert resolver.get("alpha") == 2.0


# =============================================================================
# Test Controls
# =============================================================================

class TestGalfitControls:
    def test_basic_controls(self):
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
            )

            controls.validate_inputs()

            output = str(controls)
            assert "IMAGE PARAMETERS" in output
            assert "input.fits" in output
            assert "25.0" in output

    def test_invalid_zeropoint_raises(self):
        import pytest
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_fits = tmpdir / "input.fits"
            hdu = fits.PrimaryHDU(np.zeros((100, 100)))
            hdu.writeto(input_fits)

            controls = GalfitControls(
                working_dir=tmpdir,
                input_data_image=input_fits,
                output_data_image=tmpdir / "output.fits",
                zeropoint=-1.0,
                plate_scale=(0.03, 0.03),
            )

            with pytest.raises(ValueError):
                controls.validate_inputs()


# =============================================================================
# Test Constraints Builder
# =============================================================================

class TestConstraintsBuilder:
    def test_build_constraints(self):
        config = make_base_config()
        config["constraints"] = {
            "sersic": {
                "n": "0.5 to 8",
                "q": "0.1 to 1.0",
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
            assert "sersic" in content.lower() or "n" in content
        finally:
            constraint_file.unlink(missing_ok=True)


# =============================================================================
# Test Freeze/Unfreeze
# =============================================================================

class TestFreezeUnfreeze:
    def test_freeze_all(self):
        s = Sersic(
            component_number=1,
            x_pos=GalfitParam(50.0),
            y_pos=GalfitParam(50.0),
            mag=GalfitParam(20.0),
            eff_rad=GalfitParam(10.0),
            sersic_index=GalfitParam(2.5),
            axis_ratio=GalfitParam(0.8),
            pos_angle=GalfitParam(45.0),
        )

        s.freeze_all_params()

        assert s.x_pos.freeze is True
        assert s.y_pos.freeze is True
        assert s.mag.freeze is True

    def test_unfreeze_all(self):
        s = Sersic(
            component_number=1,
            x_pos=GalfitParam(50.0, freeze=True),
            y_pos=GalfitParam(50.0, freeze=True),
            mag=GalfitParam(20.0, freeze=True),
            eff_rad=GalfitParam(10.0, freeze=True),
            sersic_index=GalfitParam(2.5, freeze=True),
            axis_ratio=GalfitParam(0.8, freeze=True),
            pos_angle=GalfitParam(45.0, freeze=True),
        )

        s.unfreeze_all_params()

        assert s.x_pos.freeze is False
        assert s.y_pos.freeze is False


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
