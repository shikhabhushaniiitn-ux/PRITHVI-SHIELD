
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from prithvi_shield.geospatial.raster_validation import (
    validate_raster,
)


def create_test_raster(path):
    """Create a small synthetic GeoTIFF for testing."""
    data = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.uint8,
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=3,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_origin(80.0, 29.0, 0.01, 0.01),
    ) as dataset:
        dataset.write(data, 1)


def test_validate_raster_returns_expected_metadata(tmp_path):
    raster_path = tmp_path / "sample.tif"
    create_test_raster(raster_path)

    result = validate_raster(raster_path)

    assert result["status"] == "valid"
    assert result["width"] == 3
    assert result["height"] == 2
    assert result["band_count"] == 1
    assert result["crs"] == "EPSG:4326"
    assert result["driver"] == "GTiff"


def test_validate_raster_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing.tif"

    with pytest.raises(FileNotFoundError):
        validate_raster(missing_path)


def test_validate_raster_rejects_file_without_crs(tmp_path):
    raster_path = tmp_path / "no_crs.tif"
    data = np.zeros((2, 3), dtype=np.uint8)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=2,
        width=3,
        count=1,
        dtype=data.dtype,
        transform=from_origin(80.0, 29.0, 0.01, 0.01),
    ) as dataset:
        dataset.write(data, 1)

    with pytest.raises(ValueError, match="CRS"):
        validate_raster(raster_path)