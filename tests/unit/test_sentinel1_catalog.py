
import io
import json
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from prithvi_shield.data.sentinel1_catalog import (
    search_sentinel1_items,
)


BBOX = (84.7, 27.5, 85.5, 28.2)
START = "2026-08-01T00:00:00Z"
END = "2026-09-01T00:00:00Z"


def fake_response(payload):
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


def test_search_sends_expected_stac_query():
    response_data = {
        "type": "FeatureCollection",
        "features": [{"id": "TEST_S1_ITEM", "properties": {}}],
    }

    with patch(
        "prithvi_shield.data.sentinel1_catalog.urlopen",
        return_value=fake_response(response_data),
    ) as mocked_urlopen:
        items = search_sentinel1_items(
            bbox=BBOX,
            start_datetime=START,
            end_datetime=END,
            limit=10,
        )

    assert len(items) == 1
    assert items[0]["id"] == "TEST_S1_ITEM"

    request = mocked_urlopen.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))

    assert payload["collections"] == ["sentinel-1-grd"]
    assert payload["bbox"] == list(BBOX)
    assert payload["datetime"] == f"{START}/{END}"
    assert payload["limit"] == 10


def test_search_follows_next_page_link():
    first_page = {
        "type": "FeatureCollection",
        "features": [{"id": "ITEM_1", "properties": {}}],
        "links": [
            {
                "rel": "next",
                "href": "https://stac.example.test/v1/search?cursor=next",
                "method": "POST",
                "body": {"cursor": "next"},
                "merge": True,
            }
        ],
    }
    second_page = {
        "type": "FeatureCollection",
        "features": [{"id": "ITEM_2", "properties": {}}],
        "links": [],
    }

    with patch(
        "prithvi_shield.data.sentinel1_catalog.urlopen",
        side_effect=[fake_response(first_page), fake_response(second_page)],
    ) as mocked_urlopen:
        items = search_sentinel1_items(
            bbox=BBOX,
            start_datetime=START,
            end_datetime=END,
            limit=10,
        )

    assert [item["id"] for item in items] == ["ITEM_1", "ITEM_2"]
    assert mocked_urlopen.call_count == 2

    second_request = mocked_urlopen.call_args_list[1].args[0]
    second_payload = json.loads(second_request.data.decode("utf-8"))

    assert second_request.full_url.endswith("?cursor=next")
    assert second_payload["collections"] == ["sentinel-1-grd"]
    assert second_payload["cursor"] == "next"


def test_search_stops_when_no_next_link_exists():
    response_data = {
        "type": "FeatureCollection",
        "features": [{"id": "ITEM_1", "properties": {}}],
        "links": [],
    }

    with patch(
        "prithvi_shield.data.sentinel1_catalog.urlopen",
        return_value=fake_response(response_data),
    ) as mocked_urlopen:
        items = search_sentinel1_items(
            bbox=BBOX,
            start_datetime=START,
            end_datetime=END,
        )

    assert len(items) == 1
    assert mocked_urlopen.call_count == 1


def test_search_removes_duplicate_items_across_pages():
    first_page = {
        "type": "FeatureCollection",
        "features": [{"id": "SAME_ITEM", "properties": {}}],
        "links": [
            {
                "rel": "next",
                "href": "https://stac.example.test/v1/search?page=2",
                "method": "POST",
                "body": {"page": 2},
                "merge": True,
            }
        ],
    }
    second_page = {
        "type": "FeatureCollection",
        "features": [
            {"id": "SAME_ITEM", "properties": {}},
            {"id": "NEW_ITEM", "properties": {}},
        ],
        "links": [],
    }

    with patch(
        "prithvi_shield.data.sentinel1_catalog.urlopen",
        side_effect=[fake_response(first_page), fake_response(second_page)],
    ):
        items = search_sentinel1_items(
            bbox=BBOX,
            start_datetime=START,
            end_datetime=END,
        )

    assert [item["id"] for item in items] == ["SAME_ITEM", "NEW_ITEM"]


def test_search_rejects_invalid_bbox():
    with pytest.raises(ValueError, match="west < east"):
        search_sentinel1_items(
            bbox=(85.5, 27.5, 84.7, 28.2),
            start_datetime=START,
            end_datetime=END,
        )


def test_search_rejects_naive_datetime():
    with pytest.raises(ValueError, match="timezone"):
        search_sentinel1_items(
            bbox=BBOX,
            start_datetime="2026-08-01T00:00:00",
            end_datetime=END,
        )


def test_search_rejects_invalid_date_order():
    with pytest.raises(ValueError, match="earlier"):
        search_sentinel1_items(
            bbox=BBOX,
            start_datetime=END,
            end_datetime=START,
        )


def test_search_rejects_invalid_limit():
    with pytest.raises(ValueError, match="positive integer"):
        search_sentinel1_items(
            bbox=BBOX,
            start_datetime=START,
            end_datetime=END,
            limit=0,
        )


def test_search_rejects_malformed_stac_response():
    with patch(
        "prithvi_shield.data.sentinel1_catalog.urlopen",
        return_value=fake_response({"type": "FeatureCollection"}),
    ):
        with pytest.raises(RuntimeError, match="features list"):
            search_sentinel1_items(
                bbox=BBOX,
                start_datetime=START,
                end_datetime=END,
            )


def test_search_rejects_invalid_next_link():
    response_data = {
        "type": "FeatureCollection",
        "features": [],
        "links": [{"rel": "next"}],
    }

    with patch(
        "prithvi_shield.data.sentinel1_catalog.urlopen",
        return_value=fake_response(response_data),
    ):
        with pytest.raises(RuntimeError, match="next-page link"):
            search_sentinel1_items(
                bbox=BBOX,
                start_datetime=START,
                end_datetime=END,
            )


def test_search_raises_if_later_page_fails():
    first_page = {
        "type": "FeatureCollection",
        "features": [{"id": "ITEM_1", "properties": {}}],
        "links": [
            {
                "rel": "next",
                "href": "https://stac.example.test/v1/search?page=2",
                "method": "POST",
                "body": {"page": 2},
                "merge": True,
            }
        ],
    }
    http_error = HTTPError(
        "https://stac.example.test/v1/search?page=2",
        503,
        "Service unavailable",
        {},
        None,
    )

    with patch(
        "prithvi_shield.data.sentinel1_catalog.urlopen",
        side_effect=[fake_response(first_page), http_error],
    ):
        with pytest.raises(RuntimeError, match="HTTP 503"):
            search_sentinel1_items(
                bbox=BBOX,
                start_datetime=START,
                end_datetime=END,
            )