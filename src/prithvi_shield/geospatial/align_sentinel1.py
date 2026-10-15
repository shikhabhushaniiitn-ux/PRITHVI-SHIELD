
"""Create AOI-only, aligned VV/VH Sentinel-1 rasters."""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window

AOI_WGS84 = (84.2, 27.6, 85.6, 28.4)
BASE = Path("data/processed/snap_test")
OUT = Path("data/processed/change_detection")
TARGET_CRS = "EPSG:32645"
PIXEL_SIZE = 10.0
NODATA = -9999.0

SCENES = {
    "pre": (
        BASE / "S1D_IW_GRDH_1SDV_20260824T001844_20260824T001909_004260_007D5D_EDEC_Cal_TC_UTM.data",
    ),
    "post": (
        BASE / "S1D_IW_GRDH_1SDV_20260905T001844_20260905T001909_004435_00837B_2F0C_Cal_TC_UTM.data",
    ),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    left, bottom, right, top = transform_bounds(
        "EPSG:4326", TARGET_CRS, *AOI_WGS84, densify_pts=21
    )
    left = np.floor(left / PIXEL_SIZE) * PIXEL_SIZE
    bottom = np.floor(bottom / PIXEL_SIZE) * PIXEL_SIZE
    right = np.ceil(right / PIXEL_SIZE) * PIXEL_SIZE
    top = np.ceil(top / PIXEL_SIZE) * PIXEL_SIZE

    width = int(round((right - left) / PIXEL_SIZE))
    height = int(round((top - bottom) / PIXEL_SIZE))
    target_transform = from_origin(left, top, PIXEL_SIZE, PIXEL_SIZE)

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": target_transform,
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    for date_label, (folder,) in SCENES.items():
        for polarization in ("VV", "VH"):
            source_path = folder / f"Sigma0_{polarization}.img"
            output_path = OUT / f"{date_label}_{polarization.lower()}_aoi_aligned.tif"

            if not source_path.exists():
                raise FileNotFoundError(f"Missing input band: {source_path}")

            with rasterio.open(source_path) as src, rasterio.open(
                output_path, "w", **profile
            ) as dst:
                for row in range(0, height, 512):
                    h = min(512, height - row)
                    for col in range(0, width, 512):
                        w = min(512, width - col)
                        window = Window(col, row, w, h)
                        tile_transform = rasterio.windows.transform(
                            window, target_transform
                        )
                        tile_bounds = rasterio.windows.bounds(
                            window, target_transform
                        )

                        # Read only the source region needed for this tile.
                        source_window = rasterio.windows.from_bounds(
                            *tile_bounds, transform=src.transform
                        ).round_offsets().round_lengths()

                        tile = np.full((h, w), NODATA, dtype=np.float32)
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=tile,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            src_nodata=src.nodata,
                            dst_transform=tile_transform,
                            dst_crs=TARGET_CRS,
                            dst_nodata=NODATA,
                            resampling=Resampling.bilinear,
                            init_dest_nodata=True,
                            warp_mem_limit=64,
                        )
                        dst.write(tile, 1, window=window)

            print(f"Created: {output_path}")

    print(f"Aligned grid: {TARGET_CRS}, {width} x {height}")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
