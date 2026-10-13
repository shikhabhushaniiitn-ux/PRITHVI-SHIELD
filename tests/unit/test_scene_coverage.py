
from shapely.geometry import box, mapping

from prithvi_shield.data.scene_coverage import (
    assess_scene_coverage,
    assess_shared_coverage,
)


AOI = (84.2, 27.6, 85.6, 28.4)


def test_full_coverage_returns_fraction_near_one():
    footprint = mapping(box(84.0, 27.4, 85.8, 28.6))

    result = assess_scene_coverage(footprint, AOI)

    assert result.intersects_aoi is True
    assert result.coverage_fraction > 0.999
    assert result.intersection_area_km2 > 0


def test_partial_coverage_returns_fraction_between_zero_and_one():
    footprint = mapping(box(84.0, 27.4, 85.0, 28.6))

    result = assess_scene_coverage(footprint, AOI)

    assert result.intersects_aoi is True
    assert 0 < result.coverage_fraction < 1


def test_no_coverage_returns_zero():
    footprint = mapping(box(86.0, 29.0, 87.0, 30.0))

    result = assess_scene_coverage(footprint, AOI)

    assert result.intersects_aoi is False
    assert result.coverage_fraction == 0


def test_shared_coverage_for_identical_footprints():
    footprint = mapping(box(84.0, 27.4, 85.8, 28.6))

    result = assess_shared_coverage(footprint, footprint, AOI)

    assert result.shared_coverage_fraction > 0.999


def test_shared_coverage_uses_common_area_only():
    first = mapping(box(84.0, 27.4, 85.0, 28.6))
    second = mapping(box(84.8, 27.4, 86.0, 28.6))

    result = assess_shared_coverage(first, second, AOI)

    assert 0 < result.shared_coverage_fraction < 1


def test_disjoint_footprints_have_zero_shared_coverage():
    first = mapping(box(84.0, 27.4, 84.5, 28.6))
    second = mapping(box(85.2, 27.4, 86.0, 28.6))

    result = assess_shared_coverage(first, second, AOI)

    assert result.shared_coverage_fraction == 0


def test_invalid_aoi_bbox_is_rejected():
    footprint = mapping(box(84.0, 27.4, 85.8, 28.6))

    try:
        assess_scene_coverage(footprint, (85.0, 28.0, 84.0, 27.0))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid AOI bbox to be rejected.")


def test_invalid_geometry_is_rejected():
    invalid_footprint = {
        "type": "Polygon",
        "coordinates": [[
            (84.0, 27.0),
            (85.0, 28.0),
            (84.0, 28.0),
            (85.0, 27.0),
            (84.0, 27.0),
        ]],
    }

    try:
        assess_scene_coverage(invalid_footprint, AOI)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid geometry to be rejected.")