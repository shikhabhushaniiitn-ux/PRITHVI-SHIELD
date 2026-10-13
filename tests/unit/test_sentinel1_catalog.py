
import io
import json
from unittest.mock import patch

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