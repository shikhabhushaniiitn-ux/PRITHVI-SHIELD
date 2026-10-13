
"""Compare Sentinel-1 catalogue coverage around the flood event.

This script searches catalogue metadata only.
It does not download satellite imagery.
"""

import sys
from pathlib import Path

# Add the project's src directory to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from prithvi_shield.data.sentinel1_catalog import search_sentinel1_items


# Provisional search boundary for the Trishuli River corridor.
# Coordinates are WGS84: west, south, east, north.
# This is not the final validated study-area boundary.
BBOX = (84.2, 27.6, 85.6, 28.4)

# Compare observations before, on, and after the proposed event date.
TIME_WINDOWS = {
    "PRE_EVENT": (
        "2026-08-10T00:00:00Z",
        "2026-08-26T00:00:00Z",
    ),
    "EVENT_DAY": (
        "2026-08-26T00:00:00Z",
        "2026-08-27T00:00:00Z",
    ),
    "POST_EVENT": (
        "2026-08-27T00:00:00Z",
        "2026-09-10T00:00:00Z",
    ),
}


def main() -> None:
    print("Sentinel-1 catalogue coverage diagnostic")
    print("Metadata search only; no imagery will be downloaded.")
    print(f"Provisional search bounding box: {BBOX}")

    for window_name, (start, end) in TIME_WINDOWS.items():
        print(f"\n{'=' * 55}")
        print(f"WINDOW: {window_name}")
        print(f"From: {start}")
        print(f"To:   {end}")

        try:
            items = search_sentinel1_items(
                bbox=BBOX,
                start_datetime=start,
                end_datetime=end,
                limit=100,
            )
        except RuntimeError as exc:
            print(f"CATALOGUE ERROR: {exc}")
            continue

        print(f"Items returned: {len(items)}")

        if not items:
            print("No items returned for this window.")
            continue

        for item in items:
            properties = item.get("properties") or {}
            geometry = item.get("geometry") or {}
            footprint_bounds = item.get("bbox")

            print(
                f"- ID: {item.get('id')}\n"
                f"  Acquired: {properties.get('datetime')}\n"
                f"  Platform: {properties.get('platform')}\n"
                f"  Polarizations: "
                f"{properties.get('sar:polarizations')}\n"
                f"  Orbit: {properties.get('sat:orbit_state')}\n"
                f"  Footprint bounds: {footprint_bounds}\n"
                f"  Geometry type: {geometry.get('type')}"
            )


if __name__ == "__main__":
    main()