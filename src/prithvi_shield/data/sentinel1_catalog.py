
"""Search the Copernicus Data Space STAC catalogue for Sentinel-1 GRD items."""

import json
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_STAC_ENDPOINT = "https://stac.dataspace.copernicus.eu/v1/search"
SENTINEL1_COLLECTION = "sentinel-1-grd"
MAX_PAGES = 1000


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
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Dates must be valid ISO 8601 timestamps.") from exc

    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Dates must include a timezone, such as Z.")

    if start >= end:
        raise ValueError("start_datetime must be earlier than end_datetime.")


def _request_json(
    url: str,
    method: str,
    payload: dict[str, Any] | None,
    headers: dict[str, str] | None,
    timeout_seconds: int | float,
) -> dict[str, Any]:
    """Send one HTTP request and parse its JSON response."""
    request_headers = {"Accept": "application/geo+json, application/json"}

    if payload is not None:
        request_headers["Content-Type"] = "application/json"

    if headers is not None:
        request_headers.update(headers)

    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    request = Request(
        url,
        data=body,
        headers=request_headers,
        method=method,
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
    except (TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "The STAC request timed out or returned invalid JSON."
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError("The STAC response must be a JSON object.")

    return result


def _get_features(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and return the features from one STAC page."""
    features = result.get("features")

    if not isinstance(features, list):
        raise RuntimeError("The STAC response has no valid features list.")

    if any(not isinstance(feature, dict) for feature in features):
        raise RuntimeError("The STAC response contains an invalid feature.")

    return features


def _get_next_link(result: dict[str, Any]) -> dict[str, Any] | None:
    """Return the STAC next-page link, if present."""
    links = result.get("links", [])

    if not isinstance(links, list):
        raise RuntimeError("The STAC response has an invalid links list.")

    next_links = [
        link
        for link in links
        if isinstance(link, dict) and link.get("rel") == "next"
    ]

    if not next_links:
        return None

    link = next_links[0]

    if not isinstance(link.get("href"), str) or not link["href"].strip():
        raise RuntimeError("The STAC next-page link has no valid href.")

    method = link.get("method", "GET")
    if not isinstance(method, str) or method.upper() not in {"GET", "POST"}:
        raise RuntimeError("The STAC next-page link uses an unsupported method.")

    headers = link.get("headers", {})
    if not isinstance(headers, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in headers.items()
    ):
        raise RuntimeError("The STAC next-page link has invalid headers.")

    body = link.get("body")
    if body is not None and not isinstance(body, dict):
        raise RuntimeError("The STAC next-page link has an invalid body.")

    merge = link.get("merge", False)
    if not isinstance(merge, bool):
        raise RuntimeError("The STAC next-page link has an invalid merge value.")

    return link


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

    `limit` is the requested page size, not a total-result cap.
    This function follows STAC next-page links and returns metadata only.
    It does not download imagery.
    """
    _validate_bbox(bbox)
    _validate_datetime_range(start_datetime, end_datetime)

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer.")

    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive.")

    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty URL.")

    original_payload: dict[str, Any] = {
        "collections": [SENTINEL1_COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{start_datetime}/{end_datetime}",
        "limit": limit,
    }

    current_url = endpoint
    current_method = "POST"
    current_payload: dict[str, Any] | None = original_payload
    current_headers: dict[str, str] = {}
    visited_requests: set[str] = set()
    all_features: list[dict[str, Any]] = []
    seen_item_ids: set[tuple[str, str]] = set()

    for page_number in range(MAX_PAGES):
        request_signature = json.dumps(
            {
                "url": current_url,
                "method": current_method,
                "payload": current_payload,
                "headers": current_headers,
            },
            sort_keys=True,
        )

        if request_signature in visited_requests:
            raise RuntimeError("The STAC catalogue returned a pagination loop.")

        visited_requests.add(request_signature)

        result = _request_json(
            url=current_url,
            method=current_method,
            payload=current_payload,
            headers=current_headers,
            timeout_seconds=timeout_seconds,
        )

        features = _get_features(result)

        for feature in features:
            item_id = feature.get("id")
            collection = feature.get("collection", SENTINEL1_COLLECTION)

            if isinstance(item_id, str) and item_id:
                identity = (str(collection), item_id)
                if identity in seen_item_ids:
                    continue
                seen_item_ids.add(identity)

            all_features.append(feature)

        next_link = _get_next_link(result)

        if next_link is None:
            return all_features

        if page_number == MAX_PAGES - 1:
            raise RuntimeError(
                f"STAC pagination exceeded the safety limit of {MAX_PAGES} pages."
            )

        current_url = urljoin(current_url, next_link["href"])
        current_method = next_link.get("method", "GET").upper()
        current_headers = next_link.get("headers", {})

        next_body = next_link.get("body")
        merge_body = next_link.get("merge", False)

        if current_method == "GET":
            current_payload = None
        elif merge_body:
            current_payload = dict(original_payload)
            if next_body is not None:
                current_payload.update(next_body)
        else:
            current_payload = next_body

    raise RuntimeError("STAC pagination ended unexpectedly.")