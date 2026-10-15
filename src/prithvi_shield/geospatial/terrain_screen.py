
"""Terrain-screen flood candidates while preserving the input validity footprint.

Output values:
    0   = valid non-candidate or candidate excluded by slope > threshold
    1   = candidate with valid slope <= threshold
    255 = original input NoData or candidate whose slope is invalid
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CANDIDATE_MASK = (
    PROJECT_ROOT / "data" / "processed" / "change_detection"
    / "candidate_both_pol_clean.tif"
)

SLOPE_RASTER = (
    PROJECT_ROOT / "data" / "processed" / "terrain"
    / "slope_aoi_30m.tif"
)

OUTPUT_MASK = (
    PROJECT_ROOT / "data" / "processed" / "change_detection"
    / "candidate_terrain_screened_10deg.tif"
)

SLOPE_THRESHOLD_DEGREES = 10.0
OUTPUT_NODATA = 255


def main():
    for path in (CANDIDATE_MASK, SLOPE_RASTER):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    with rasterio.open(CANDIDATE_MASK) as candidate_src:
        candidate = candidate_src.read(1)
        profile = candidate_src.profile.copy()
        candidate_nodata = candidate_src.nodata

        if candidate_nodata is None:
            candidate_valid = np.ones(candidate.shape, dtype=bool)
        else:
            candidate_valid = candidate != candidate_nodata

        candidate_transform = candidate_src.transform
        candidate_crs = candidate_src.crs
        candidate_shape = candidate.shape

    if candidate_crs is None:
        raise ValueError("Candidate mask has no CRS.")

    slope_aligned = np.full(
        candidate_shape, np.nan, dtype=np.float32
    )

    with rasterio.open(SLOPE_RASTER) as slope_src:
        if slope_src.crs is None:
            raise ValueError("Slope raster has no CRS.")

        reproject(
            source=rasterio.band(slope_src, 1),
            destination=slope_aligned,
            src_transform=slope_src.transform,
            src_crs=slope_src.crs,
            src_nodata=slope_src.nodata,
            dst_transform=candidate_transform,
            dst_crs=candidate_crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
            init_dest_nodata=True,
        )

    slope_valid = (
        np.isfinite(slope_aligned)
        & (slope_aligned >= 0)
        & (slope_aligned <= 90)
    )

    is_candidate = candidate_valid & (candidate == 1)
    is_non_candidate = candidate_valid & (candidate == 0)

    keep = (
        is_candidate
        & slope_valid
        & (slope_aligned <= SLOPE_THRESHOLD_DEGREES)
    )

    excluded_steep = (
        is_candidate
        & slope_valid
        & (slope_aligned > SLOPE_THRESHOLD_DEGREES)
    )

    candidates_without_slope = is_candidate & ~slope_valid
    non_candidates_without_slope = is_non_candidate & ~slope_valid

    # Begin with NoData, then preserve valid non-candidates even if
    # slope data is unavailable. Unknown-slope candidates remain NoData.
    output = np.full(
        candidate_shape, OUTPUT_NODATA, dtype=np.uint8
    )

    output[is_non_candidate] = 0
    output[excluded_steep] = 0
    output[keep] = 1

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=OUTPUT_NODATA,
        compress="deflate",
    )

    OUTPUT_MASK.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(OUTPUT_MASK, "w", **profile) as dst:
        dst.write(output, 1)
        dst.set_band_description(
            1,
            "Terrain-screened flood candidates; "
            f"slope <= {SLOPE_THRESHOLD_DEGREES} degrees",
        )

    print("Terrain screening complete.")
    print(f"Input valid pixels: {int(candidate_valid.sum()):,}")
    print(f"Original candidate pixels: {int(is_candidate.sum()):,}")
    print(f"Retained candidate pixels: {int(keep.sum()):,}")
    print(f"Excluded on steep terrain: {int(excluded_steep.sum()):,}")
    print(
        "Candidates with invalid slope (left as NoData): "
        f"{int(candidates_without_slope.sum()):,}"
    )
    print(
        "Non-candidates with invalid slope (preserved as 0): "
        f"{int(non_candidates_without_slope.sum()):,}"
    )
    print(f"Slope threshold: {SLOPE_THRESHOLD_DEGREES:.1f} degrees")
    print(f"Output saved: {OUTPUT_MASK}")

    print(
        "Note: terrain-screened candidates are not confirmed "
        "flood extent."
    )


if __name__ == "__main__":
    main()
