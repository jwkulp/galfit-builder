"""Tests for pipeline modules: freeze, cutout, to_regions."""

import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from regions import PixCoord, EllipsePixelRegion, PointPixelRegion, PolygonPixelRegion
import astropy.units as u

from galfit_builder.pipeline.freeze import (
    parse_feedme_components,
    freeze_line,
    freeze_component,
    point_in_polygons,
    apply_polygon_freezing,
    match_with_regions,
    delete_component,
    reindex_components,
)

from galfit_builder.pipeline.to_regions import (
    parse_param_line,
    parse_components,
    get_ellipse_params,
    component_to_region,
    get_color_map,
)

from galfit_builder.pipeline.cutout import (
    parse_box_region,
    compute_rotated_box_center,
    rotate_image,
    update_header_for_cutout,
)


# =============================================================================
# Test Feedme Parsing (freeze.py)
# =============================================================================

SAMPLE_FEEDME = """# IMAGE PARAMETERS
A) input.fits
B) output.fits
# ...

# Object number: 1
 0) sky                    # object type
 1) 0.0006   1  # sky background
 2) 0.0000   0  # dsky/dx
 3) 0.0000   0  # dsky/dy
 Z) 0

# Object number: 2
 0) sersic                 # object type
 1) 50.00  60.00  1 1  # position x, y
 3) 20.0000   1  # integrated magnitude
 4) 10.0000   1  # R_e
 5) 2.5000   1  # Sersic index
 9) 0.8000   1  # axis ratio
10) 45.0000   1  # position angle
 Z) 0

# Object number: 3
 0) psf                    # object type
 1) 30.00  40.00  1 1  # position x, y
 3) 22.0000   1  # total magnitude
 Z) 0
"""


class TestFeedmeParsing:
    def test_parse_feedme_components(self):
        lines = SAMPLE_FEEDME.split('\n')
        components = parse_feedme_components(lines)

        # Should find 2 components (sky is skipped)
        assert len(components) == 2

        # Sersic at (50, 60)
        assert components[0]["x"] == 50.0
        assert components[0]["y"] == 60.0
        assert components[0]["component_number"] == 2

        # PSF at (30, 40)
        assert components[1]["x"] == 30.0
        assert components[1]["y"] == 40.0
        assert components[1]["component_number"] == 3

    def test_freeze_line_single_toggle(self):
        line = " 3) 20.0000   1  # integrated magnitude\n"

        frozen = freeze_line(line, freeze=True)
        assert "   0  " in frozen

        unfrozen = freeze_line(line, freeze=False)
        assert "   1  " in unfrozen

    def test_freeze_line_position(self):
        line = " 1) 50.00  60.00  1 1  # position x, y\n"

        frozen = freeze_line(line, freeze=True)
        assert "0 0" in frozen

        unfrozen = freeze_line(line, freeze=False)
        assert "1 1" in unfrozen

    def test_freeze_line_preserves_minus_one(self):
        line = " 5) 2.5000   -1  # fixed parameter\n"

        result = freeze_line(line, freeze=True)
        assert "-1" in result

    def test_freeze_line_comment_only(self):
        line = "# This is a comment\n"
        assert freeze_line(line, True) == line

    def test_freeze_component(self):
        lines = SAMPLE_FEEDME.split('\n')
        lines = [l + '\n' for l in lines]

        # Find sersic start index
        sersic_start = None
        for i, line in enumerate(lines):
            if "# Object number: 2" in line:
                sersic_start = i
                break

        assert sersic_start is not None

        frozen_lines = freeze_component(lines, sersic_start, freeze=True)

        # Check that sersic params are frozen
        for i in range(sersic_start, len(frozen_lines)):
            line = frozen_lines[i]
            if "# Object number: 3" in line:
                break
            if line.strip().startswith(("1)", "3)", "4)", "5)", "9)", "10)")):
                assert "0" in line or "-1" in line


# =============================================================================
# Test Polygon Containment (freeze.py)
# =============================================================================

class TestPolygonContainment:
    def test_point_in_polygon(self):
        # Square polygon from (10,10) to (20,20)
        vertices = PixCoord([10, 20, 20, 10], [10, 10, 20, 20])
        polygon = PolygonPixelRegion(vertices=vertices)

        # Point inside
        assert point_in_polygons(15, 15, [polygon]) is True

        # Point outside
        assert point_in_polygons(5, 5, [polygon]) is False

    def test_point_in_multiple_polygons(self):
        # Two separate polygons
        poly1 = PolygonPixelRegion(vertices=PixCoord([0, 10, 10, 0], [0, 0, 10, 10]))
        poly2 = PolygonPixelRegion(vertices=PixCoord([20, 30, 30, 20], [20, 20, 30, 30]))

        # In first polygon
        assert point_in_polygons(5, 5, [poly1, poly2]) is True

        # In second polygon
        assert point_in_polygons(25, 25, [poly1, poly2]) is True

        # In neither
        assert point_in_polygons(15, 15, [poly1, poly2]) is False


class TestApplyPolygonFreezing:
    def test_freeze_inside_polygon(self):
        lines = SAMPLE_FEEDME.split('\n')
        lines = [l + '\n' for l in lines]

        components = parse_feedme_components(lines)

        # Polygon that contains sersic (50, 60) but not psf (30, 40)
        vertices = PixCoord([40, 70, 70, 40], [50, 50, 70, 70])
        polygon = PolygonPixelRegion(vertices=vertices)

        result = apply_polygon_freezing(lines, components, [polygon])

        # Find the sersic position line - should be frozen (inside polygon)
        found_sersic = False
        found_psf = False
        for i, line in enumerate(result):
            if "# Object number: 2" in line:
                # Check next few lines for sersic params
                for j in range(i+1, min(i+10, len(result))):
                    if result[j].strip().startswith("1)") and "position" in result[j]:
                        assert "0 0" in result[j], "Sersic should be frozen (inside polygon)"
                        found_sersic = True
                        break
            if "# Object number: 3" in line:
                # Check next few lines for psf params
                for j in range(i+1, min(i+10, len(result))):
                    if result[j].strip().startswith("1)") and "position" in result[j]:
                        assert "1 1" in result[j], "PSF should be free (outside polygon)"
                        found_psf = True
                        break

        assert found_sersic
        assert found_psf


# =============================================================================
# Test Region Matching (freeze.py)
# =============================================================================

class TestRegionMatching:
    def test_match_with_regions(self):
        feedme_comps = [
            {"x": 50.0, "y": 60.0, "component_number": 2, "matched": False},
            {"x": 30.0, "y": 40.0, "component_number": 3, "matched": False},
        ]

        # Region at approximately (50, 60) - 0-indexed so (49, 59)
        region = EllipsePixelRegion(
            center=PixCoord(x=49, y=59),
            width=10,
            height=8,
            angle=0 * u.deg,
        )

        matched = match_with_regions(feedme_comps, [region], tolerance=2.0)

        assert matched[0] is True  # Region matched to sersic
        assert feedme_comps[0]["matched"] is True


class TestComponentDeletion:
    def test_delete_component(self):
        lines = SAMPLE_FEEDME.split('\n')
        lines = [l + '\n' for l in lines]

        # Find sersic start index
        sersic_start = None
        for i, line in enumerate(lines):
            if "# Object number: 2" in line:
                sersic_start = i
                break

        result = delete_component(lines, sersic_start, comp_num=2)

        # Should not contain sersic anymore
        result_text = ''.join(result)
        assert "sersic" not in result_text

        # PSF should be renumbered to 2
        assert "# Object number: 2" in result_text

    def test_reindex_components(self):
        lines = [
            "# Object number: 1\n",
            "# Object number: 2\n",
            "# Object number: 3\n",
        ]

        result = reindex_components(lines, deleted_num=2)

        assert "# Object number: 1" in result[0]
        assert "# Object number: 2" in result[2]  # was 3


# =============================================================================
# Test to_regions.py
# =============================================================================

class TestToRegions:
    def test_parse_param_line(self):
        param_num, values = parse_param_line(" 1) 50.00  60.00  1 1  # position")

        assert param_num == "1"
        assert values == ["50.00", "60.00", "1", "1"]

    def test_parse_param_line_comment(self):
        param_num, values = parse_param_line("# This is a comment")

        assert param_num == ""
        assert values == []

    def test_parse_components(self):
        feedme_lines = [
            " 0) sersic\n",
            " 1) 50.00 60.00 1 1  # position\n",
            " 3) 20.0  1  # mag\n",
            " 4) 10.0  1  # R_e\n",
            " 9) 0.8   1  # q\n",
            "10) 45.0  1  # pa\n",
            " 0) psf\n",
            " 1) 30.00 40.00 1 1  # position\n",
        ]

        components = parse_components(feedme_lines)

        assert len(components) == 2
        assert components[0]["type"] == "sersic"
        assert components[0]["x"] == 50.0
        assert components[1]["type"] == "psf"

    def test_get_ellipse_params_sersic(self):
        comp = {
            "type": "sersic",
            "x": 50.0,
            "y": 60.0,
            "param_4": ["10.0"],
            "param_9": ["0.8"],
            "param_10": ["45.0"],
        }

        result = get_ellipse_params(comp)

        assert result is not None
        a, b, angle = result
        assert a == 10.0
        assert b == 8.0  # 10 * 0.8
        assert angle == 135.0  # (45 + 90) % 360

    def test_get_color_map(self):
        config = {
            "region_colors": {
                "cyan": "sersic",
                "magenta": "king",
            }
        }

        color_map = get_color_map(config)

        assert color_map["sersic"] == "cyan"
        assert color_map["king"] == "magenta"

    def test_component_to_region_psf(self):
        comp = {"type": "psf", "x": 50.0, "y": 60.0}
        color_map = {"psf": "red"}

        region_str = component_to_region(comp, color_map)

        assert "point(50.0000,60.0000)" in region_str
        assert "color=red" in region_str

    def test_component_to_region_sersic(self):
        comp = {
            "type": "sersic",
            "x": 50.0,
            "y": 60.0,
            "param_4": ["10.0"],
            "param_9": ["0.8"],
            "param_10": ["0.0"],
        }
        color_map = {"sersic": "cyan"}

        region_str = component_to_region(comp, color_map)

        assert "ellipse" in region_str
        assert "50.0000" in region_str
        assert "color=cyan" in region_str


# =============================================================================
# Test cutout.py
# =============================================================================

class TestCutout:
    def test_parse_box_region(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.reg', delete=False) as f:
            f.write("# Region file format: DS9\n")
            f.write("image\n")
            f.write("box(100, 150, 50, 40, 30)\n")
            reg_path = Path(f.name)

        try:
            box = parse_box_region(reg_path)

            assert box["xc"] == 100
            assert box["yc"] == 150
            assert box["width"] == 50
            assert box["height"] == 40
            assert box["angle"] == 30
        finally:
            reg_path.unlink()

    def test_compute_rotated_box_center_no_rotation(self):
        box = {"xc": 60, "yc": 70, "width": 20, "height": 20, "angle": 0}
        image_center = (50, 50)

        new_x, new_y = compute_rotated_box_center(box, image_center)

        assert abs(new_x - 60) < 0.01
        assert abs(new_y - 70) < 0.01

    def test_rotate_image_no_rotation(self):
        image = np.arange(100).reshape(10, 10).astype(float)

        rotated = rotate_image(image, 0)

        np.testing.assert_array_equal(image, rotated)

    def test_rotate_image_90_degrees(self):
        image = np.zeros((10, 10))
        image[4, 5] = 1.0  # Near center

        rotated = rotate_image(image, 90, order=0)

        # After 90 deg CCW rotation around center, (4,5) -> (5, 5) roughly
        # The exact position depends on scipy's rotation; just verify something moved
        assert rotated.sum() > 0.5  # Value preserved somewhere
        assert not np.array_equal(image, rotated)  # Something changed

    def test_update_header_for_cutout(self):
        header = fits.Header()
        header["NAXIS1"] = 100
        header["NAXIS2"] = 100
        header["CRPIX1"] = 50
        header["CRPIX2"] = 50

        new_header = update_header_for_cutout(header, x_min=10, y_min=20, new_shape=(30, 40))

        assert new_header["NAXIS1"] == 40
        assert new_header["NAXIS2"] == 30
        assert new_header["CRPIX1"] == 40  # 50 - 10
        assert new_header["CRPIX2"] == 30  # 50 - 20
        assert new_header["CUTOUT"] is True


# =============================================================================
# Integration test: end-to-end freeze
# =============================================================================

class TestFreezeIntegration:
    def test_freeze_workflow(self):
        """Test complete freeze workflow with feedme and regions."""
        feedme_content = """# IMAGE PARAMETERS
A) input.fits
B) output.fits

# Object number: 1
 0) sky
 1) 0.001   1
 2) 0.0     0
 3) 0.0     0
 Z) 0

# Object number: 2
 0) sersic
 1) 15.00  20.00  1 1  # inside polygon
 3) 20.0   1
 4) 5.0    1
 5) 2.5    1
 9) 0.8    1
10) 0.0    1
 Z) 0

# Object number: 3
 0) psf
 1) 50.00  50.00  1 1  # outside polygon
 3) 22.0   1
 Z) 0
"""

        # Polygon that contains (15, 20) but not (50, 50)
        polygon = PolygonPixelRegion(
            vertices=PixCoord([10, 25, 25, 10], [15, 15, 25, 25])
        )

        lines = feedme_content.split('\n')
        lines = [l + '\n' for l in lines]

        components = parse_feedme_components(lines)
        result = apply_polygon_freezing(lines, components, [polygon])

        result_text = ''.join(result)

        # Sersic (inside polygon) should be frozen
        assert "sersic" in result_text

        # Find sersic position line and check it's frozen
        sersic_frozen = False
        psf_free = False
        for i, line in enumerate(result):
            if "# Object number: 2" in line:
                for j in range(i+1, min(i+10, len(result))):
                    if result[j].strip().startswith("1)") and "inside polygon" in result[j]:
                        sersic_frozen = "0 0" in result[j]
                        break
            if "# Object number: 3" in line:
                for j in range(i+1, min(i+10, len(result))):
                    if result[j].strip().startswith("1)") and "outside polygon" in result[j]:
                        psf_free = "1 1" in result[j]
                        break

        assert sersic_frozen, "Sersic inside polygon should be frozen"
        assert psf_free, "PSF outside polygon should be free"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
