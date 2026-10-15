
"""Combine terrain and water-occurrence candidate masks.

Encoding:
    0   = valid non-candidate or candidate rejected by a screen
    1   = candidate passing both screens
    255 = NoData or a candidate that cannot be classified by both screens
"""

from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "processed" / "change_detection"

TERRAIN_MASK = BASE / "candidate_terrain_screened_10deg.tif"

WATER_MASKS = {
    50: BASE / "candidate_water_screened_50pct.tif",
    80: BASE / "candidate_water_screened_80pct.tif",
}

OUTPUTS = {
    50: BASE / "candidate_combined_screened_50pct.tif",
    80: BASE / "candidate_combined_screened_80pct.tif",
}

NODATA = 255


def valid_mask(data, nodata):
    """Return pixels that are not the declared NoData value."""
    if nodata is None:
        return np.ones(data.shape, dtype=bool)
    return data != nodata


def main():
    with rasterio.open(TERRAIN_MASK) as terrain_src:
        terrain = terrain_src.read(1)
        profile = terrain_src.profile.copy()
        terrain_nodata = terrain_src.nodata
        terrain_valid = valid_mask(terrain, terrain_nodata)

        if terrain_src.crs is None:
            raise ValueError("Terrain mask has no CRS.")

        terrain_candidates = terrain_valid & (terrain == 1)
        terrain_non_candidates = terrain_valid & (terrain == 0)

        for threshold, water_path in WATER_MASKS.items():
            if not water_path.is_file():
                raise FileNotFoundError(
                    f"Water-screened mask not found: {water_path}"
                )

            with rasterio.open(water_path) as water_src:
                if (
                    water_src.crs != terrain_src.crs
                    or water_src.transform != terrain_src.transform
                    or water_src.width != terrain_src.width
                    or water_src.height != terrain_src.height
                ):
                    raise ValueError(
                        f"Grid mismatch between terrain and "
                        f"{water_path.name}"
                    )

                water = water_src.read(1)
                water_valid = valid_mask(water, water_src.nodata)

            water_candidates = water_valid & (water == 1)
            water_non_candidates = water_valid & (water == 0)

            # A valid non-candidate in either screen cannot become a
            # combined candidate, so preserve it as 0 even if the other
            # screen is NoData.
            known_non_candidate = (
                terrain_non_candidates | water_non_candidates
            )

            # A candidate is retained only if both masks explicitly
            # classify it as a candidate.
            combined_candidates = (
                terrain_candidates
                & water_candidates
            )

            # If neither mask establishes a non-candidate, but one or
            # both screens are unknown, leave the result unclassified.
            combined_valid = (
                known_non_candidate | combined_candidates
            )

            result = np.full(
                terrain.shape, NODATA, dtype=np.uint8
            )
            result[combined_valid] = 0
            result[combined_candidates] = 1

            output_path = OUTPUTS[threshold]
            output_profile = profile.copy()
            output_profile.update(
                driver="GTiff",
                count=1,
                dtype="uint8",
                nodata=NODATA,
                compress="deflate",
            )

            with rasterio.open(
                output_path, "w", **output_profile
            ) as dst:
                dst.write(result, 1)
                dst.set_band_description(
                    1,
                    "Combined terrain and water-occurrence "
                    f"candidate mask; occurrence below {threshold}%",
                )

            counts = {
                value: int((result == value).sum())
                for value in (0, 1, NODATA)
            }

            print(f"\nWater occurrence threshold: {threshold}%")
            print(f"Non-candidate pixels (0): {counts[0]:,}")
            print(f"Candidate pixels (1): {counts[1]:,}")
            print(f"NoData pixels (255): {counts[NODATA]:,}")
            print(f"Total pixels: {sum(counts.values()):,}")
            print(f"Nominal candidate area: {counts[1] * 0.0001:.4f} km²")
            print(f"Saved: {output_path}")

    print(
        "\nCombined outputs are experimental candidate masks, "
        "not confirmed flood extent."
    )


if __name__ == "__main__":
    main()
