
from datetime import datetime, timezone

import pytest

from prithvi_shield.data.observation import SatelliteObservation


def make_observation(**overrides):
    values = {
        "product_id": "TEST_PRODUCT_001",
        "collection_id": "sentinel-1-grd",
        "acquired_at": datetime(
            2026, 8, 15, 5, 30, tzinfo=timezone.utc
        ),
        "platform": "Sentinel-1",
        "product_type": "GRD",
        "polarizations": ("vv", "vh"),
        "catalog_url": (
            "https://stac.dataspace.copernicus.eu/"
            "v1/collections/sentinel-1-grd"
        ),
        "source_url": None,
        "orbit_direction": "ascending",
    }

    values.update(overrides)
    return SatelliteObservation(**values)


def test_observation_normalizes_metadata():
    observation = make_observation()

    assert observation.polarizations == ("VV", "VH")
    assert observation.orbit_direction == "ASCENDING"


def test_observation_serializes_to_json_friendly_dict():
    observation = make_observation()

    result = observation.to_dict()

    assert result["product_id"] == "TEST_PRODUCT_001"
    assert result["collection_id"] == "sentinel-1-grd"
    assert result["acquired_at"].endswith("+00:00")
    assert result["polarizations"] == ["VV", "VH"]


def test_observation_rejects_naive_datetime():
    with pytest.raises(ValueError, match="timezone-aware"):
        make_observation(
            acquired_at=datetime(2026, 8, 15, 5, 30)
        )


def test_observation_rejects_invalid_polarization():
    with pytest.raises(ValueError, match="Polarizations"):
        make_observation(polarizations=("XX",))