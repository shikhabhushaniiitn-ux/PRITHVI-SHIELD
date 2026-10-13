
"""Screen Sentinel-1 scenes using metadata and AOI footprint coverage.

This script does not download imagery, detect flooding, or estimate damage.
The AOI is a provisional WGS84 rectangle, not a validated study boundary.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from prithvi_shield.data.scene_coverage import (  # noqa: E402
    assess_scene_coverage,
    assess_shared_coverage,
)
from prithvi_shield.data.scene_pairing import (  # noqa: E402
    SceneCandidate,
    assess_scene_pair,
)
from prithvi_shield.data.sentinel1_catalog import (  # noqa: E402
    search_sentinel1_items,
)


# Provisional rectangle only. Validate the actual study boundary later.
BBOX = (84.2, 27.6, 85.6, 28.4)
PROJECTED_CRS = "EPSG:32645"

PRE_EVENT_WINDOW = (
    "2026-08-10T00:00:00Z",
    "2026-08-26T00:00:00Z",
)
POST_EVENT_WINDOW = (
    "2026-08-27T00:00:00Z",
    "2026-09-10T00:00:00Z",
)

# Screening settings: configurable project choices, not universal standards.
EVENT_REFERENCE = datetime(2026, 8, 26, tzinfo=timezone.utc)
MAX_DAYS_APART = 30
CATALOGUE_LIMIT = 100

MIN_INDIVIDUAL_AOI_COVERAGE = 0.10
MIN_SHARED_AOI_COVERAGE = 0.10
PREFERRED_COVERAGE_THRESHOLD = 0.80
MAX_EVENT_DISTANCE_DAYS_FOR_SCORE = 30.0

OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "sentinel1_scene_pair_screening.json"
)


def parse_datetime(value: str) -> datetime:
    """Parse a timezone-aware ISO 8601 timestamp."""
    if not isinstance(value, str):
        raise ValueError("STAC acquisition datetime must be a string.")

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"Invalid STAC acquisition datetime: {value!r}"
        ) from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            "STAC acquisition datetime must include a timezone."
        )

    return parsed


def stac_item_to_candidate(item: dict[str, Any]) -> SceneCandidate:
    """Convert a STAC feature into validated scene metadata."""
    if not isinstance(item, dict):
        raise ValueError("STAC item must be a dictionary.")

    properties = item.get("properties") or {}
    item_id = item.get("id")
    acquired = properties.get("datetime")
    orbit = properties.get("sat:orbit_state")
    polarizations = properties.get("sar:polarizations")
    bbox = item.get("bbox")

    if not item_id:
        raise ValueError("STAC item has no ID.")
    if not acquired:
        raise ValueError(f"STAC item {item_id} has no acquisition datetime.")
    if not orbit:
        raise ValueError(f"STAC item {item_id} has no orbit direction.")
    if not isinstance(polarizations, list) or not polarizations:
        raise ValueError(
            f"STAC item {item_id} has no valid polarization list."
        )
    if not isinstance(bbox, list) or len(bbox) < 4:
        raise ValueError(f"STAC item {item_id} has no valid bounding box.")

    try:
        footprint_bbox = tuple(float(value) for value in bbox[:4])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"STAC item {item_id} has an invalid bounding box."
        ) from exc

    return SceneCandidate(
        product_id=str(item_id),
        acquisition_datetime=parse_datetime(acquired),
        orbit_direction=str(orbit),
        polarizations=frozenset(str(value) for value in polarizations),
        footprint_bbox=footprint_bbox,
    )


def search_window(
    window_name: str,
    window: tuple[str, str],
) -> tuple[
    list[SceneCandidate],
    list[dict[str, str]],
    dict[str, dict[str, Any]],
]:
    """Search a time window and assess each item's footprint coverage."""
    start, end = window
    print(f"\nSearching {window_name}: {start} to {end}")

    items = search_sentinel1_items(
        bbox=BBOX,
        start_datetime=start,
        end_datetime=end,
        limit=CATALOGUE_LIMIT,
    )
    print(f"STAC items returned: {len(items)}")

    candidates: list[SceneCandidate] = []
    rejected_items: list[dict[str, str]] = []
    coverage_by_product: dict[str, dict[str, Any]] = {}

    for item in items:
        item_id = (
            str(item.get("id", "UNKNOWN"))
            if isinstance(item, dict)
            else "UNKNOWN"
        )

        try:
            candidate = stac_item_to_candidate(item)
        except (TypeError, ValueError) as exc:
            rejected_items.append(
                {"product_id": item_id, "reason": str(exc)}
            )
            continue

        geometry = item.get("geometry")
        if not isinstance(geometry, dict):
            rejected_items.append(
                {
                    "product_id": candidate.product_id,
                    "reason": (
                        "STAC item has no GeoJSON footprint geometry; "
                        "coverage cannot be assessed."
                    ),
                }
            )
            continue

        try:
            coverage = assess_scene_coverage(
                geometry,
                BBOX,
                projected_crs=PROJECTED_CRS,
            )
        except (TypeError, ValueError) as exc:
            rejected_items.append(
                {
                    "product_id": candidate.product_id,
                    "reason": f"Footprint coverage assessment failed: {exc}",
                }
            )
            continue

        coverage_by_product[candidate.product_id] = {
            "coverage": coverage.to_dict(),
            "geometry": geometry,
        }
        candidates.append(candidate)

    print(f"Items usable for coverage screening: {len(candidates)}")
    print(f"Items rejected: {len(rejected_items)}")

    for candidate in candidates:
        percent = coverage_by_product[candidate.product_id][
            "coverage"
        ]["coverage_percent"]
        print(f"  AOI coverage: {candidate.product_id}: {percent:.2f}%")

    return candidates, rejected_items, coverage_by_product


def assess_pair_with_coverage(
    pre_scene: SceneCandidate,
    post_scene: SceneCandidate,
    pre_coverage: dict[str, Any],
    post_coverage: dict[str, Any],
) -> dict[str, Any]:
    """Assess metadata, individual coverage, shared coverage, and ranking."""
    metadata = assess_scene_pair(
        pre_scene,
        post_scene,
        max_days_apart=MAX_DAYS_APART,
    )

    shared = assess_shared_coverage(
        pre_coverage["geometry"],
        post_coverage["geometry"],
        BBOX,
        projected_crs=PROJECTED_CRS,
    )

    pre_fraction = float(
        pre_coverage["coverage"]["coverage_fraction"]
    )
    post_fraction = float(
        post_coverage["coverage"]["coverage_fraction"]
    )
    shared_fraction = float(shared.shared_coverage_fraction)

    reasons = list(metadata.reasons)

    # Minimum coverage rules prevent tiny overlaps being accepted.
    if pre_fraction < MIN_INDIVIDUAL_AOI_COVERAGE:
        reasons.append(
            "Pre-event AOI coverage is below the configured minimum "
            f"of {MIN_INDIVIDUAL_AOI_COVERAGE:.0%}."
        )

    if post_fraction < MIN_INDIVIDUAL_AOI_COVERAGE:
        reasons.append(
            "Post-event AOI coverage is below the configured minimum "
            f"of {MIN_INDIVIDUAL_AOI_COVERAGE:.0%}."
        )

    if shared_fraction < MIN_SHARED_AOI_COVERAGE:
        reasons.append(
            "Shared AOI coverage is below the configured minimum "
            f"of {MIN_SHARED_AOI_COVERAGE:.0%}."
        )

    coverage_passes = (
        pre_fraction >= MIN_INDIVIDUAL_AOI_COVERAGE
        and post_fraction >= MIN_INDIVIDUAL_AOI_COVERAGE
        and shared_fraction >= MIN_SHARED_AOI_COVERAGE
    )

    accepted = bool(metadata.candidate and coverage_passes)

    preferred = (
        accepted
        and pre_fraction >= PREFERRED_COVERAGE_THRESHOLD
        and post_fraction >= PREFERRED_COVERAGE_THRESHOLD
        and shared_fraction >= PREFERRED_COVERAGE_THRESHOLD
    )

    if preferred:
        status = "preferred"
        reasons.append(
            "Preferred screening thresholds passed: both individual "
            "coverages and shared coverage meet the configured "
            f"{PREFERRED_COVERAGE_THRESHOLD:.0%} threshold."
        )
    elif accepted:
        status = "partial_coverage"
        reasons.append(
            "Metadata and minimum coverage checks passed, but the pair "
            "does not meet every preferred coverage threshold."
        )
    else:
        status = "rejected"
        if not metadata.candidate:
            reasons.append(
                "The pair failed the existing metadata compatibility checks."
            )

    # A transparent heuristic ranking score from 0 to 100.
    # Coverage dominates; proximity to the reference event date contributes
    # less. This score is not a probability or model confidence.
    pre_event_distance = abs(
        (EVENT_REFERENCE - pre_scene.acquisition_datetime).total_seconds()
    ) / 86400.0
    post_event_distance = abs(
        (post_scene.acquisition_datetime - EVENT_REFERENCE).total_seconds()
    ) / 86400.0
    total_event_distance = pre_event_distance + post_event_distance

    temporal_score = max(
        0.0,
        1.0
        - total_event_distance / MAX_EVENT_DISTANCE_DAYS_FOR_SCORE,
    )

    weakest_individual_coverage = min(pre_fraction, post_fraction)

    ranking_score = 100.0 * (
        0.70 * shared_fraction
        + 0.20 * weakest_individual_coverage
        + 0.10 * temporal_score
    )

    result = metadata.to_dict()
    result.update(
        {
            "candidate": accepted,
            "pair_status": status,
            "pre_event_aoi_coverage_fraction": round(pre_fraction, 6),
            "pre_event_aoi_coverage_percent": round(pre_fraction * 100, 2),
            "post_event_aoi_coverage_fraction": round(post_fraction, 6),
            "post_event_aoi_coverage_percent": round(post_fraction * 100, 2),
            "shared_aoi_coverage": shared.to_dict(),
            "pre_event_distance_from_reference_days": round(
                pre_event_distance, 2
            ),
            "post_event_distance_from_reference_days": round(
                post_event_distance, 2
            ),
            "temporal_score": round(temporal_score, 4),
            "ranking_score": round(ranking_score, 2),
            "ranking_score_note": (
                "Heuristic 0-100 screening score: 70% shared coverage, "
                "20% weaker individual coverage, 10% temporal proximity. "
                "Not a probability, accuracy estimate, or flood score."
            ),
            "reasons": reasons,
        }
    )
    return result


def main() -> None:
    """Run screening and write a transparent JSON report."""
    print("PRITHVI-SHIELD: Sentinel-1 scene-pair screening")
    print("Metadata and STAC-footprint geometry only.")
    print("No imagery will be downloaded.")
    print(f"Provisional search bbox: {BBOX}")
    print(f"Coverage projection: {PROJECTED_CRS}")
    print(f"Maximum acquisition gap: {MAX_DAYS_APART} days")
    print(
        "Minimum individual coverage: "
        f"{MIN_INDIVIDUAL_AOI_COVERAGE:.0%}"
    )
    print(f"Minimum shared coverage: {MIN_SHARED_AOI_COVERAGE:.0%}")
    print(
        "Preferred individual and shared coverage: "
        f"{PREFERRED_COVERAGE_THRESHOLD:.0%}"
    )

    try:
        pre_scenes, pre_rejected, pre_coverage = search_window(
            "PRE_EVENT", PRE_EVENT_WINDOW
        )
        post_scenes, post_rejected, post_coverage = search_window(
            "POST_EVENT", POST_EVENT_WINDOW
        )
    except (RuntimeError, ValueError) as exc:
        print(f"\nCatalogue or metadata error: {exc}")
        print("No complete screening report was generated.")
        return

    assessments: list[dict[str, Any]] = []

    for pre_scene in pre_scenes:
        for post_scene in post_scenes:
            if (
                pre_scene.acquisition_datetime
                >= post_scene.acquisition_datetime
            ):
                continue

            try:
                assessment = assess_pair_with_coverage(
                    pre_scene,
                    post_scene,
                    pre_coverage[pre_scene.product_id],
                    post_coverage[post_scene.product_id],
                )
            except (TypeError, ValueError) as exc:
                assessment = {
                    "pre_event_product_id": pre_scene.product_id,
                    "post_event_product_id": post_scene.product_id,
                    "candidate": False,
                    "pair_status": "rejected",
                    "ranking_score": 0.0,
                    "reasons": [
                        f"Shared coverage assessment failed: {exc}"
                    ],
                }

            assessments.append(assessment)

    candidates = [
        item for item in assessments if item.get("candidate", False)
    ]
    candidates.sort(
        key=lambda item: item.get("ranking_score", 0.0),
        reverse=True,
    )

    for rank, pair in enumerate(candidates, start=1):
        pair["rank"] = rank

    preferred_count = sum(
        pair["pair_status"] == "preferred" for pair in candidates
    )
    partial_count = sum(
        pair["pair_status"] == "partial_coverage" for pair in candidates
    )
    rejected_pair_count = sum(
        pair.get("pair_status") == "rejected" for pair in assessments
    )

    report = {
        "project": "PRITHVI-SHIELD",
        "purpose": (
            "Sentinel-1 metadata and footprint-based scene-pair screening"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalogue_endpoint": (
            "https://stac.dataspace.copernicus.eu/v1/search"
        ),
        "collection": "sentinel-1-grd",
        "search_bbox_wgs84": list(BBOX),
        "aoi_status": (
            "Provisional rectangular search envelope; not a validated "
            "study-area polygon."
        ),
        "coverage_method": {
            "source": "STAC item GeoJSON footprint geometry",
            "projected_crs": PROJECTED_CRS,
            "individual_coverage": (
                "Scene footprint intersection area divided by total AOI area."
            ),
            "shared_coverage": (
                "Area covered by both footprints inside the AOI divided "
                "by total AOI area."
            ),
        },
        "screening_thresholds": {
            "minimum_individual_aoi_coverage": (
                MIN_INDIVIDUAL_AOI_COVERAGE
            ),
            "minimum_shared_aoi_coverage": MIN_SHARED_AOI_COVERAGE,
            "preferred_individual_and_shared_coverage": (
                PREFERRED_COVERAGE_THRESHOLD
            ),
            "note": (
                "These are configurable screening choices, not universal "
                "scientific standards. Validate them for the final AOI "
                "and analysis objective."
            ),
        },
        "ranking_method": {
            "event_reference_datetime_utc": EVENT_REFERENCE.isoformat(),
            "weights": {
                "shared_coverage": 0.70,
                "weaker_individual_coverage": 0.20,
                "temporal_proximity": 0.10,
            },
            "note": (
                "Heuristic ranking score from 0 to 100; not a probability, "
                "model confidence, flood detection, or damage estimate."
            ),
        },
        "pre_event_window": list(PRE_EVENT_WINDOW),
        "post_event_window": list(POST_EVENT_WINDOW),
        "max_days_apart": MAX_DAYS_APART,
        "catalogue_limit_per_window": CATALOGUE_LIMIT,
        "pagination_note": (
            "The current catalogue client does not follow STAC next links. "
            "This report only assesses the items returned by each request; "
            "catalogue coverage may be incomplete."
        ),
        "interpretation_note": (
            "Candidate status means metadata checks and configured footprint "
            "coverage thresholds passed. It does not establish valid pixel "
            "coverage, co-registration, radiometric comparability, flood "
            "extent, or damage. Different pair combinations are not "
            "necessarily independent observations."
        ),
        "pre_event_items_returned": len(pre_scenes) + len(pre_rejected),
        "post_event_items_returned": len(post_scenes) + len(post_rejected),
        "pre_event_items_usable_for_coverage": len(pre_scenes),
        "post_event_items_usable_for_coverage": len(post_scenes),
        "pre_event_scene_coverage": {
            product_id: entry["coverage"]
            for product_id, entry in pre_coverage.items()
        },
        "post_event_scene_coverage": {
            product_id: entry["coverage"]
            for product_id, entry in post_coverage.items()
        },
        "pre_event_rejected_items": pre_rejected,
        "post_event_rejected_items": post_rejected,
        "chronologically_valid_pairs_assessed": len(assessments),
        "candidate_pair_count": len(candidates),
        "preferred_pair_count": preferred_count,
        "partial_coverage_pair_count": partial_count,
        "rejected_pair_count": rejected_pair_count,
        "candidate_pairs_ranked": candidates,
        "all_pair_assessments": assessments,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("SCREENING SUMMARY")
    print(f"Pre-event scenes: {len(pre_scenes)}")
    print(f"Post-event scenes: {len(post_scenes)}")
    print(f"Chronologically valid pairs assessed: {len(assessments)}")
    print(f"Accepted candidate pairs: {len(candidates)}")
    print(f"  Preferred pairs: {preferred_count}")
    print(f"  Partial-coverage pairs: {partial_count}")
    print(f"  Rejected pair assessments: {rejected_pair_count}")
    print(f"JSON report: {OUTPUT_PATH}")

    if candidates:
        print("\nRanked candidate pairs:")
        for pair in candidates:
            shared_percent = pair["shared_aoi_coverage"][
                "shared_coverage_percent"
            ]
            print(
                f"\n#{pair['rank']} [{pair['pair_status']}] "
                f"score={pair['ranking_score']:.2f}/100"
            )
            print(f"  Pre:  {pair['pre_event_product_id']}")
            print(f"  Post: {pair['post_event_product_id']}")
            print(
                f"  Gap: {pair.get('days_between_acquisitions', 'N/A')} days"
            )
            print(
                f"  Coverage pre/post/shared: "
                f"{pair['pre_event_aoi_coverage_percent']:.2f}% / "
                f"{pair['post_event_aoi_coverage_percent']:.2f}% / "
                f"{shared_percent:.2f}%"
            )
    else:
        print("\nNo pairs passed the configured screening thresholds.")

    print(
        "\nReminder: candidate status is not a flood-detection result."
    )


if __name__ == "__main__":
    main()