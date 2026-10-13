
"""Search the Copernicus Data Space STAC catalogue for Sentinel-1 GRD items."""

import json
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_STAC_ENDPOINT = "https://stac.dataspace.copernicus.eu/v1/search"
SENTINEL1_COLLECTION = "sentinel-1-grd"


def _validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    """Validate a WGS84 bounding box: west, south, east, north."""
    if len(bbox) != 4:
        raise ValueError("bbox must contain west, south, east, and north.")

    west, south, east, north = bbox

    if not all(-180 <= lon <= 180 for lon in (west, east)):
        raise ValueError("Longitude values must be between -180 and 180.")

    if not all(-90 <= lat <= 90 for lat in (south, north)):
        raise ValueError("Latitude values must be between -90 and 90.")

    if west >= east or south >= north:
        raise ValueError("bbox must have west < east and south < north.")


def _validate_datetime_range(start_datetime: str, end_datetime: str) -> None:
    """Require valid ISO 8601 timestamps in chronological order."""
    try:
        start = datetime.fromisoformat(start_datetime.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_datetime.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Dates must be valid ISO 8601 timestamps.") from exc

    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Dates must include a timezone, such as Z.")

    if start >= end:
        raise ValueError("start_datetime must be earlier than end_datetime.")


def search_sentinel1_items(
    bbox: tuple[float, float, float, float],
    start_datetime: str,
    end_datetime: str,
    limit: int = 20,
    endpoint: str = DEFAULT_STAC_ENDPOINT,
    timeout_seconds: int = 30,
) -> list[dict[str, Any]]:
    """
    Search the STAC catalogue for Sentinel-1 GRD observations.

    bbox order: west longitude, south latitude,
                east longitude, north latitude.

    This function searches metadata only. It does not download imagery.
    """
    _validate_bbox(bbox)
    _validate_datetime_range(start_datetime, end_datetime)

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer.")

    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    payload = {
        "collections": [SENTINEL1_COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{start_datetime}/{end_datetime}",
        "limit": limit,
    }

    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(
            f"STAC catalogue returned HTTP {exc.code}."
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            "Could not connect to the STAC catalogue. "
            "Check your internet connection and try again."
        ) from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "The STAC request timed out or returned invalid JSON."
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError("The STAC response must be a JSON object.")

    features = result.get("features")

    if not isinstance(features, list):
        raise RuntimeError("The STAC response has no valid features list.")

    return features