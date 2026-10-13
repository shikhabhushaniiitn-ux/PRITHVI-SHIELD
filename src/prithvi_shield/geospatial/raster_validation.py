
from pathlib import Path

import rasterio


def validate_raster(path: str | Path) -> dict:
    """
    Read basic metadata from a raster file and validate
    that it has usable dimensions, bands, and a CRS.

    This function checks file structure and metadata only.
    It does not validate scientific suitability for flood mapping.
    """
    raster_path = Path(path)

    if not raster_path.exists():
        raise FileNotFoundError(
            f"Raster file not found: {raster_path}"
        )

    if not raster_path.is_file():
        raise ValueError(
            f"Raster path is not a file: {raster_path}"
        )

    try:
        with rasterio.open(raster_path) as dataset:
            if dataset.width <= 0 or dataset.height <= 0:
                raise ValueError(
                    "Raster must have positive width and height."
                )

            if dataset.count <= 0:
                raise ValueError(
                    "Raster must contain at least one band."
                )

            if dataset.crs is None:
                raise ValueError(
                    "Raster has no coordinate reference system (CRS)."
                )

            bounds = dataset.bounds

            return {
                "path": str(raster_path),
                "width": dataset.width,
                "height": dataset.height,
                "band_count": dataset.count,
                "crs": dataset.crs.to_string(),
                "bounds": {
                    "left": bounds.left,
                    "bottom": bounds.bottom,
                    "right": bounds.right,
                    "top": bounds.top,
                },
                "resolution": list(dataset.res),
                "driver": dataset.driver,
                "status": "valid",
            }

    except rasterio.errors.RasterioError as exc:
        raise ValueError(
            f"Unable to read raster file: {raster_path}"
        ) from exc