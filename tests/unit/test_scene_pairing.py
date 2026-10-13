
"""Unit tests for Sentinel-1 scene-pair screening."""

from datetime import datetime, timezone

import pytest

from prithvi_shield.data.scene_pairing import (
    SceneCandidate,
    assess_scene_pair,
    bounding_boxes_overlap,
)


def make_scene(
    product_id: str,
    acquisition_datetime: datetime,
    *,
    orbit_direction: str = "DESCENDING",
    polarizations: frozenset[str] = frozenset({"VV", "VH"}),
    footprint_bbox: tuple[float, float, float, float] = (
        83.0, 27.0, 86.0, 29.0
    ),
) -> SceneCandidate:
    """Build a scene fixture for tests."""
    return SceneCandidate(
        product_id=product_id,
        acquisition_datetime=acquisition_datetime,
        orbit_direction=orbit_direction,
        polarizations=polarizations,
        footprint_bbox=footprint_bbox,
    )


def test_matching_scenes_are_candidates():
    pre = make_scene(
        "PRE_001",
        datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc),
    )
    post = make_scene(
        "POST_001",
        datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc),
    )

    result = assess_scene_pair(pre, post)

    assert result.candidate is True
    assert result.orbit_direction_matches is True
    assert result.polarizations_match is True
    assert result.footprints_overlap is True
    assert result.days_between_acquisitions == 12
    assert result.pre_event_product_id == "PRE_001"
    assert result.post_event_product_id == "POST_001"


def test_different_orbit_directions_are_rejected():
    pre = make_scene(
        "PRE_002",
        datetime(2026, 8, 21, tzinfo=timezone.utc),
        orbit_direction="ASCENDING",
    )
    post = make_scene(
        "POST_002",
        datetime(2026, 9, 2, tzinfo=timezone.utc),
        orbit_direction="DESCENDING",
    )

    result = assess_scene_pair(pre, post)

    assert result.candidate is False
    assert result.orbit_direction_matches is False
    assert any("Orbit directions differ" in reason for reason in result.reasons)


def test_different_polarizations_are_rejected():
    pre = make_scene(
        "PRE_003",
        datetime(2026, 8, 21, tzinfo=timezone.utc),
        polarizations=frozenset({"VV", "VH"}),
    )
    post = make_scene(
        "POST_003",
        datetime(2026, 9, 2, tzinfo=timezone.utc),
        polarizations=frozenset({"VV"}),
    )

    result = assess_scene_pair(pre, post)

    assert result.candidate is False
    assert result.polarizations_match is False


def test_non_overlapping_footprints_are_rejected():
    pre = make_scene(
        "PRE_004",
        datetime(2026, 8, 21, tzinfo=timezone.utc),
        footprint_bbox=(80.0, 20.0, 81.0, 21.0),
    )
    post = make_scene(
        "POST_004",
        datetime(2026, 9, 2, tzinfo=timezone.utc),
        footprint_bbox=(83.0, 27.0, 86.0, 29.0),
    )

    result = assess_scene_pair(pre, post)

    assert result.candidate is False
    assert result.footprints_overlap is False


def test_pair_beyond_time_limit_is_rejected():
    pre = make_scene(
        "PRE_005",
        datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    post = make_scene(
        "POST_005",
        datetime(2026, 9, 15, tzinfo=timezone.utc),
    )

    result = assess_scene_pair(pre, post, max_days_apart=30)

    assert result.candidate is False
    assert result.days_between_acquisitions == 45
    assert any("maximum allowed" in reason for reason in result.reasons)


def test_pre_event_must_be_earlier_than_post_event():
    pre = make_scene(
        "PRE_006",
        datetime(2026, 9, 2, tzinfo=timezone.utc),
    )
    post = make_scene(
        "POST_006",
        datetime(2026, 8, 21, tzinfo=timezone.utc),
    )

    with pytest.raises(ValueError, match="must be earlier"):
        assess_scene_pair(pre, post)


def test_bounding_boxes_that_overlap_are_detected():
    first = (83.0, 27.0, 85.0, 29.0)
    second = (84.0, 28.0, 86.0, 30.0)

    assert bounding_boxes_overlap(first, second) is True


def test_touching_bounding_boxes_do_not_have_positive_area_overlap():
    first = (83.0, 27.0, 84.0, 28.0)
    second = (84.0, 27.0, 85.0, 28.0)

    assert bounding_boxes_overlap(first, second) is False


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError, match="timezone"):
        make_scene(
            "PRE_007",
            datetime(2026, 8, 21, 12, 0),
        )


def test_invalid_bbox_is_rejected():
    with pytest.raises(ValueError, match="positive area"):
        make_scene(
            "PRE_008",
            datetime(2026, 8, 21, tzinfo=timezone.utc),
            footprint_bbox=(86.0, 29.0, 83.0, 27.0),
        )


def test_assessment_can_be_serialized_to_dict():
    pre = make_scene(
        "PRE_009",
        datetime(2026, 8, 21, tzinfo=timezone.utc),
    )
    post = make_scene(
        "POST_009",
        datetime(2026, 9, 2, tzinfo=timezone.utc),
    )

    result = assess_scene_pair(pre, post)
    output = result.to_dict()

    assert output["candidate"] is True
    assert output["pre_event_product_id"] == "PRE_009"
    assert isinstance(output["reasons"], list)

