
"""Screen returned Sentinel-1 STAC items for candidate pre/post pairs.

Metadata screening only: this script does not download imagery or detect
floods. Results depend on the items returned by the current catalogue
client and are not a guarantee of complete catalogue coverage.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from prithvi_shield.data.scene_pairing import (  # noqa: E402
    SceneCandidate,
    assess_scene_pair,
)
from prithvi_shield.data.sentinel1_catalog import (  # noqa: E402
    search_sentinel1_items,
)


# Provisional search envelope, not the final validated study-area boundary.
BBOX = (84.2, 27.6, 85.6, 28.4)

PRE_EVENT_WINDOW = (
    "2026-08-10T00:00:00Z",
    "2026-08-26T00:00:00Z",
)
POST_EVENT_WINDOW = (
    "2026-08-27T00:00:00Z",
    "2026-09-10T00:00:00Z",
)

MAX_DAYS_APART = 30
CATALOGUE_LIMIT = 100
OUTPUT_PATH = PROJECT_ROOT / "reports" / "sentinel1_scene_pair_screening.json"


def parse_datetime(value: str) -> datetime:
    """Parse a timezone-aware ISO 8601 timestamp."""
    if not isinstance(value, str):
        raise ValueError("STAC acquisition datetime must be a string.")

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if parsed.tzinfo is None:
        raise ValueError("STAC acquisition datetime must include a timezone.")

    return parsed


def stac_item_to_candidate(item: dict[str, Any]) -> SceneCandidate:
    """Convert one STAC feature into validated scene metadata."""
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
        raise ValueError(f"STAC item {item_id} has no valid polarization list.")

    if not isinstance(bbox, list) or len(bbox) < 4:
        raise ValueError(f"STAC item {item_id} has no valid bounding box.")

    # STAC bbox may contain 4 or 6 values. We use the first four
    # coordinates: west, south, east, north.
    footprint_bbox = tuple(float(value) for value in bbox[:4])

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
) -> tuple[list[SceneCandidate], list[dict[str, str]]]:
    """Search one time window and report items that cannot be converted."""
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

    for item in items:
        try:
            candidates.append(stac_item_to_candidate(item))
        except (TypeError, ValueError) as exc:
            rejected_items.append(
                {
                    "product_id": str(item.get("id", "UNKNOWN")),
                    "reason": str(exc),
                }
            )

    print(f"Items converted: {len(candidates)}")
    print(f"Items rejected for metadata issues: {len(rejected_items)}")

    return candidates, rejected_items


def main() -> None:
    """Run screening and write a JSON report."""
    print("PRITHVI-SHIELD: Sentinel-1 scene-pair screening")
    print("Metadata only. No imagery will be downloaded.")
    print(f"Provisional search bbox: {BBOX}")
    print(f"Maximum acquisition gap: {MAX_DAYS_APART} days")

    try:
        pre_scenes, pre_rejected = search_window(
            "PRE_EVENT", PRE_EVENT_WINDOW
        )
        post_scenes, post_rejected = search_window(
            "POST_EVENT", POST_EVENT_WINDOW
        )
    except (RuntimeError, ValueError) as exc:
        print(f"\nCatalogue or metadata error: {exc}")
        print("No complete screening report was generated.")
        return

    assessments = []

    for pre_scene in pre_scenes:
        for post_scene in post_scenes:
            # Ignore combinations where the acquisition order is invalid.
            if (
                pre_scene.acquisition_datetime
                >= post_scene.acquisition_datetime
            ):
                continue

            assessment = assess_scene_pair(
                pre_scene,
                post_scene,
                max_days_apart=MAX_DAYS_APART,
            )

            # Save every chronologically valid pair, including rejected
            # pairs, so the report can explain the screening decisions.
            assessments.append(assessment.to_dict())

    candidates = [
        item for item in assessments if item["candidate"]
    ]

    report = {
        "project": "PRITHVI-SHIELD",
        "purpose": "Sentinel-1 metadata-based scene-pair screening",
        "generated_at_utc": datetime.now().astimezone().isoformat(),
        "catalogue_endpoint": (
            "https://stac.dataspace.copernicus.eu/v1/search"
        ),
        "collection": "sentinel-1-grd",
        "search_bbox_wgs84": list(BBOX),
        "pre_event_window": list(PRE_EVENT_WINDOW),
        "post_event_window": list(POST_EVENT_WINDOW),
        "max_days_apart": MAX_DAYS_APART,
        "catalogue_limit_per_window": CATALOGUE_LIMIT,
        "pagination_note": (
            "The current catalogue client does not follow STAC next links. "
            "This report assesses only the items returned by each request; "
            "catalogue coverage may therefore be incomplete."
        ),
        "interpretation_note": (
            "A candidate passed metadata checks only. This does not confirm "
            "study-area coverage, geometric co-registration, radiometric "
            "comparability, flood extent, or damage."
        ),
        "pre_event_items_returned": len(pre_scenes) + len(pre_rejected),
        "post_event_items_returned": len(post_scenes) + len(post_rejected),
        "pre_event_items_converted": len(pre_scenes),
        "post_event_items_converted": len(post_scenes),
        "pre_event_rejected_items": pre_rejected,
        "post_event_rejected_items": post_rejected,
        "chronologically_valid_pairs_assessed": len(assessments),
        "candidate_pair_count": len(candidates),
        "candidate_pairs": candidates,
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
    print(f"Candidate pairs: {len(candidates)}")
    print(f"JSON report: {OUTPUT_PATH}")

    if candidates:
        print("\nCandidate pairs:")
        for pair in candidates:
            print(
                f"- {pair['pre_event_product_id']}\n"
                f"  -> {pair['post_event_product_id']}\n"
                f"  Gap: {pair['days_between_acquisitions']} days"
            )
    else:
        print("\nNo candidate pairs passed all metadata checks.")

    print("\nRemember: candidate status is not a flood-detection result.")


if __name__ == "__main__":
    main()

