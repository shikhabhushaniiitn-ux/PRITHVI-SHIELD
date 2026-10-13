
"""Screen pre-event and post-event Sentinel-1 scene pairs.

This module evaluates metadata compatibility only. It does not download
imagery, calculate flood extent, or confirm that a scene pair is suitable
for quantitative change detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class SceneCandidate:
    """Minimal metadata required to screen a Sentinel-1 scene."""

    product_id: str
    acquisition_datetime: datetime
    orbit_direction: str
    polarizations: frozenset[str]
    footprint_bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.product_id.strip():
            raise ValueError("product_id must not be empty.")

        if self.acquisition_datetime.tzinfo is None:
            raise ValueError(
                "acquisition_datetime must include timezone information."
            )

        direction = self.orbit_direction.upper()
        if direction not in {"ASCENDING", "DESCENDING"}:
            raise ValueError(
                "orbit_direction must be ASCENDING or DESCENDING."
            )
        object.__setattr__(self, "orbit_direction", direction)

        normalized_polarizations = frozenset(
            polarization.upper() for polarization in self.polarizations
        )
        if not normalized_polarizations:
            raise ValueError("At least one polarization is required.")

        if not normalized_polarizations.issubset(
            {"VV", "VH", "HH", "HV"}
        ):
            raise ValueError("Unsupported polarization value.")

        object.__setattr__(
            self, "polarizations", normalized_polarizations
        )

        if len(self.footprint_bbox) != 4:
            raise ValueError(
                "footprint_bbox must be (west, south, east, north)."
            )

        west, south, east, north = self.footprint_bbox

        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise ValueError("Footprint longitudes must be within [-180, 180].")

        if not (-90 <= south <= 90 and -90 <= north <= 90):
            raise ValueError("Footprint latitudes must be within [-90, 90].")

        if west >= east or south >= north:
            raise ValueError("Footprint bbox must have positive area.")


@dataclass(frozen=True)
class ScenePairAssessment:
    """Metadata screening result for a pre/post scene pair."""

    pre_event_product_id: str
    post_event_product_id: str
    orbit_direction_matches: bool
    polarizations_match: bool
    footprints_overlap: bool
    days_between_acquisitions: int
    candidate: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "pre_event_product_id": self.pre_event_product_id,
            "post_event_product_id": self.post_event_product_id,
            "orbit_direction_matches": self.orbit_direction_matches,
            "polarizations_match": self.polarizations_match,
            "footprints_overlap": self.footprints_overlap,
            "days_between_acquisitions": self.days_between_acquisitions,
            "candidate": self.candidate,
            "reasons": list(self.reasons),
        }


def bounding_boxes_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    """Return whether two WGS84 bounding boxes have positive-area overlap."""
    first_west, first_south, first_east, first_north = first
    second_west, second_south, second_east, second_north = second

    return (
        first_west < second_east
        and first_east > second_west
        and first_south < second_north
        and first_north > second_south
    )


def assess_scene_pair(
    pre_event: SceneCandidate,
    post_event: SceneCandidate,
    *,
    max_days_apart: int = 30,
) -> ScenePairAssessment:
    """Assess whether two scenes are metadata-compatible candidates.

    The pre-event scene must be acquired before the post-event scene.
    Candidate status requires matching orbit direction, matching
    polarization sets, overlapping bounding boxes, and an acquisition
    gap within max_days_apart.

    Bounding-box overlap is only a coarse check. It does not prove that
    the actual scene geometries cover the study area.
    """
    if max_days_apart < 1:
        raise ValueError("max_days_apart must be at least 1.")

    if pre_event.acquisition_datetime >= post_event.acquisition_datetime:
        raise ValueError(
            "The pre-event acquisition must be earlier than the post-event acquisition."
        )

    elapsed_seconds = (
        post_event.acquisition_datetime - pre_event.acquisition_datetime
    ).total_seconds()
    days_between = int(elapsed_seconds // 86400)

    orbit_matches = (
        pre_event.orbit_direction == post_event.orbit_direction
    )
    polarizations_match = (
        pre_event.polarizations == post_event.polarizations
    )
    footprints_overlap = bounding_boxes_overlap(
        pre_event.footprint_bbox,
        post_event.footprint_bbox,
    )
    within_time_limit = days_between <= max_days_apart

    reasons: list[str] = []

    if not orbit_matches:
        reasons.append("Orbit directions differ.")

    if not polarizations_match:
        reasons.append("Polarization sets differ.")

    if not footprints_overlap:
        reasons.append("Footprint bounding boxes do not overlap.")

    if not within_time_limit:
        reasons.append(
            f"Acquisitions are {days_between} whole days apart; "
            f"maximum allowed is {max_days_apart}."
        )

    if not reasons:
        reasons.append(
            "Metadata checks passed; geometry, co-registration, "
            "and radiometric suitability remain unverified."
        )

    candidate = (
        orbit_matches
        and polarizations_match
        and footprints_overlap
        and within_time_limit
    )

    return ScenePairAssessment(
        pre_event_product_id=pre_event.product_id,
        post_event_product_id=post_event.product_id,
        orbit_direction_matches=orbit_matches,
        polarizations_match=polarizations_match,
        footprints_overlap=footprints_overlap,
        days_between_acquisitions=days_between,
        candidate=candidate,
        reasons=tuple(reasons),
    )
