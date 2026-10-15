
"""Remove small isolated regions from the candidate-change mask."""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import sieve

BASE = Path("data/processed/change_detection")
INPUT = BASE / "candidate_both_pol_raw.tif"
OUTPUT = BASE / "candidate_both_pol_clean.tif"

MIN_PATCH_PIXELS = 9
NODATA = 255


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input mask not found: {INPUT}")

    with rasterio.open(INPUT) as src:
        mask = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    valid = mask != nodata
    binary = (mask == 1).astype(np.uint8)

    # Sieve only the binary candidate mask. The valid-data mask is restored
    # afterward so nodata areas are never converted into non-candidates.
    cleaned_binary = sieve(
        binary,
        size=MIN_PATCH_PIXELS,
        connectivity=8,
        mask=valid,
    )

    cleaned = np.full(mask.shape, NODATA, dtype=np.uint8)
    cleaned[valid] = 0
    cleaned[valid & (cleaned_binary == 1)] = 1

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=NODATA,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    with rasterio.open(OUTPUT, "w", **profile) as dst:
        dst.write(cleaned, 1)

    before = int(np.count_nonzero(valid & (mask == 1)))
    after = int(np.count_nonzero(valid & (cleaned == 1)))
    pixel_area_m2 = abs(profile["transform"].a * profile["transform"].e)
    area_km2 = after * pixel_area_m2 / 1_000_000

    print("Cleaned mask:", OUTPUT)
    print("Minimum patch size:", MIN_PATCH_PIXELS, "pixels")
    print("Candidates before:", before)
    print("Candidates after:", after)
    print("Removed candidates:", before - after)
    print(f"Remaining candidate pixel area: {area_km2:.2f} km²")
    print("This remains a candidate mask, not confirmed flood extent.")


if __name__ == "__main__":
    main()
