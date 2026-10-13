"""Focused tests for Sentinel-1 scene-pair screening and ranking."""

from datetime import timedelta
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

# Make the project root and src directory importable during pytest.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

for directory in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import scripts.screen_sentinel1_pairs as screening
from prithvi_shield.data.scene_pairing import SceneCandidate


def run_assessment(
    monkeypatch,
    *,
    pre_fraction=0.90,
    post_fraction=0.90,
    shared_fraction=0.90,
    metadata_candidate=True,
):
    """Run pair screening with controlled coverage and metadata results."""

    pre_datetime = screening.EVENT_REFERENCE - timedelta(days=2)
    post_datetime = screening.EVENT_REFERENCE + timedelta(days=3)

    pre_scene = SceneCandidate(
        product_id="TEST_PRE_SCENE",
        acquisition_datetime=pre_datetime,
        orbit_direction="DESCENDING",
        polarizations=frozenset({"VV", "VH"}),
        footprint_bbox=(84.2, 27.6, 85.6, 28.4),
    )

    post_scene = SceneCandidate(
        product_id="TEST_POST_SCENE",
        acquisition_datetime=post_datetime,
        orbit_direction="DESCENDING",
        polarizations=frozenset({"VV", "VH"}),
        footprint_bbox=(84.2, 27.6, 85.6, 28.4),
    )

    metadata = SimpleNamespace(
        candidate=metadata_candidate,
        reasons=("Mock metadata assessment.",),
        to_dict=lambda: {
            "pre_event_product_id": "TEST_PRE_SCENE",
            "post_event_product_id": "TEST_POST_SCENE",
            "candidate": metadata_candidate,
            "reasons": ["Mock metadata assessment."],
        },
    )

    shared = SimpleNamespace(
        shared_coverage_fraction=shared_fraction,
        to_dict=lambda: {
            "shared_coverage_fraction": shared_fraction,
        },
    )

    monkeypatch.setattr(
        screening,
        "assess_scene_pair",
        lambda *args, **kwargs: metadata,
    )

    monkeypatch.setattr(
        screening,
        "assess_shared_coverage",
        lambda *args, **kwargs: shared,
    )

    pre_coverage = {
        "geometry": object(),
        "coverage": {"coverage_fraction": pre_fraction},
    }

    post_coverage = {
        "geometry": object(),
        "coverage": {"coverage_fraction": post_fraction},
    }

    result = screening.assess_pair_with_coverage(
        pre_scene,
        post_scene,
        pre_coverage,
        post_coverage,
    )

    return result


@pytest.mark.parametrize(
    ("pre", "post", "shared", "expected_status"),
    [
        (0.90, 0.85, 0.82, "preferred"),
        (0.50, 0.70, 0.40, "partial_coverage"),
        (0.05, 0.90, 0.90, "rejected"),
        (0.90, 0.90, 0.05, "rejected"),
    ],
)
def test_pair_status_matches_coverage_thresholds(
    monkeypatch, pre, post, shared, expected_status
):
    result = run_assessment(
        monkeypatch,
        pre_fraction=pre,
        post_fraction=post,
        shared_fraction=shared,
    )

    assert result["pair_status"] == expected_status
    assert result["candidate"] is (expected_status != "rejected")


def test_exact_preferred_threshold_is_accepted(monkeypatch):
    result = run_assessment(
        monkeypatch,
        pre_fraction=0.80,
        post_fraction=0.80,
        shared_fraction=0.80,
    )

    assert result["pair_status"] == "preferred"
    assert result["candidate"] is True


def test_exact_minimum_coverage_threshold_is_accepted(monkeypatch):
    result = run_assessment(
        monkeypatch,
        pre_fraction=0.10,
        post_fraction=0.10,
        shared_fraction=0.10,
    )

    assert result["pair_status"] == "partial_coverage"
    assert result["candidate"] is True


def test_incompatible_metadata_rejects_pair(monkeypatch):
    result = run_assessment(
        monkeypatch,
        pre_fraction=0.95,
        post_fraction=0.95,
        shared_fraction=0.95,
        metadata_candidate=False,
    )

    assert result["pair_status"] == "rejected"
    assert result["candidate"] is False
    assert any(
        "metadata compatibility" in reason.lower()
        for reason in result["reasons"]
    )


def test_ranking_score_matches_documented_formula(monkeypatch):
    pre_fraction = 0.90
    post_fraction = 0.80
    shared_fraction = 0.85

    result = run_assessment(
        monkeypatch,
        pre_fraction=pre_fraction,
        post_fraction=post_fraction,
        shared_fraction=shared_fraction,
    )

    # The reference event is 2 days after the pre-scene and
    # 3 days before the post-scene, so total distance is 5 days.
    expected_temporal_score = max(
        0.0,
        1.0 - 5.0 / screening.MAX_EVENT_DISTANCE_DAYS_FOR_SCORE,
    )

    expected_score = round(
        100.0
        * (
            0.70 * shared_fraction
            + 0.20 * min(pre_fraction, post_fraction)
            + 0.10 * expected_temporal_score
        ),
        2,
    )

    assert result["temporal_score"] == round(
        expected_temporal_score, 4
    )
    assert result["ranking_score"] == expected_score


def test_ranking_score_is_bounded_and_not_a_probability(monkeypatch):
    result = run_assessment(
        monkeypatch,
        pre_fraction=0.90,
        post_fraction=0.85,
        shared_fraction=0.80,
    )

    assert 0.0 <= result["ranking_score"] <= 100.0
    assert "not a probability" in result["ranking_score_note"].lower()


def test_coverage_percentages_are_reported_correctly(monkeypatch):
    result = run_assessment(
        monkeypatch,
        pre_fraction=0.756,
        post_fraction=0.834,
        shared_fraction=0.700,
    )

    assert result["pre_event_aoi_coverage_percent"] == 75.6
    assert result["post_event_aoi_coverage_percent"] == 83.4
