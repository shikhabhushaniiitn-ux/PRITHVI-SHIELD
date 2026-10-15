
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, transform as window_transform
from rasterio.warp import reproject


ROOT = Path(__file__).resolve().parents[3]

WATER_PATH = (
    ROOT / "data" / "raw" / "water"
    / "occurrence_80E_30N_v1_5_2024.tif"
)

CANDIDATE_PATH = (
    ROOT / "data" / "processed" / "change_detection"
    / "candidate_both_pol_clean.tif"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "change_detection"

ALIGNED_WATER = OUTPUT_DIR / "water_occurrence_aligned_10m.tif"

THRESHOLDS = (50, 80)
WINDOW_SIZE = 512
NODATA = 255


def main():
    if not WATER_PATH.exists():
        raise FileNotFoundError(f"Water raster not found: {WATER_PATH}")
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(f"Candidate mask not found: {CANDIDATE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(CANDIDATE_PATH) as candidate_src, \
         rasterio.open(WATER_PATH) as water_src:

        if candidate_src.crs is None or water_src.crs is None:
            raise ValueError("Both rasters must have a defined CRS.")

        width = candidate_src.width
        height = candidate_src.height

        print(f"Candidate CRS: {candidate_src.crs}")
        print(f"Water CRS: {water_src.crs}")
        print(f"Output grid: {width} x {height}")
        print("Resampling: nearest neighbour")

        profile = candidate_src.profile.copy()
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

        output_paths = {
            threshold: OUTPUT_DIR
            / f"candidate_water_screened_{threshold}pct.tif"
            for threshold in THRESHOLDS
        }

        # Counters track the results over the full grid.
        counts = {
            threshold: {
                "original": 0,
                "retained": 0,
                "excluded": 0,
                "missing_water": 0,
            }
            for threshold in THRESHOLDS
        }

        with rasterio.open(ALIGNED_WATER, "w", **profile) as water_dst:
            water_dst.set_band_description(
                1, "JRC water occurrence percentage, aligned to candidate grid"
            )

            with rasterio.open(output_paths[50], "w", **profile) as out50, \
                 rasterio.open(output_paths[80], "w", **profile) as out80:

                out50.set_band_description(
                    1, "Candidate pixels with water occurrence below 50%"
                )
                out80.set_band_description(
                    1, "Candidate pixels with water occurrence below 80%"
                )

                for row in range(0, height, WINDOW_SIZE):
                    for col in range(0, width, WINDOW_SIZE):
                        h = min(WINDOW_SIZE, height - row)
                        w = min(WINDOW_SIZE, width - col)
                        window = Window(col, row, w, h)

                        candidate = candidate_src.read(1, window=window)
                        if candidate_src.nodata is None:
                            candidate_valid = np.ones(
                                candidate.shape, dtype=bool
                            )
                        else:
                            candidate_valid = (
                                candidate != candidate_src.nodata
                            )

                        candidate_pixels = (
                            candidate_valid & (candidate == 1)
                        )

                        occurrence = np.full(
                            (h, w), NODATA, dtype=np.uint8
                        )

                        dst_transform = window_transform(
                            window, candidate_src.transform
                        )

                        reproject(
                            source=rasterio.band(water_src, 1),
                            destination=occurrence,
                            src_transform=water_src.transform,
                            src_crs=water_src.crs,
                            src_nodata=255,
                            dst_transform=dst_transform,
                            dst_crs=candidate_src.crs,
                            dst_nodata=NODATA,
                            resampling=Resampling.nearest,
                            init_dest_nodata=True,
                        )

                        # JRC occurrence percentages are 0-100.
                        # Treat 255 and values above 100 as invalid.
                        occurrence_valid = occurrence <= 100
                        occurrence_valid &= candidate_valid

                        water_dst.write(
                            occurrence, 1, window=window
                        )

                        for threshold, output in (
                            (50, out50),
                            (80, out80),
                        ):
                            retained = (
                                candidate_pixels
                                & occurrence_valid
                                & (occurrence < threshold)
                            )

                            excluded = (
                                candidate_pixels
                                & occurrence_valid
                                & (occurrence >= threshold)
                            )

                            missing = (
                                candidate_pixels & ~occurrence_valid
                            )

                            result = np.full(
                                (h, w), NODATA, dtype=np.uint8
                            )
                            result[candidate_valid & occurrence_valid] = 0
                            result[retained] = 1

                            output.write(result, 1, window=window)

                            counts[threshold]["original"] += int(
                                candidate_pixels.sum()
                            )
                            counts[threshold]["retained"] += int(
                                retained.sum()
                            )
                            counts[threshold]["excluded"] += int(
                                excluded.sum()
                            )
                            counts[threshold]["missing_water"] += int(
                                missing.sum()
                            )

                    print(f"Processed rows through {row + h:,}/{height:,}")

    print("\n=== WATER OCCURRENCE SCREENING RESULTS ===")
    print(f"Aligned occurrence raster: {ALIGNED_WATER}")

    for threshold in THRESHOLDS:
        result = counts[threshold]
        retained = result["retained"]
        area_km2 = retained * 0.0001

        print(f"\nThreshold: {threshold}%")
        print(f"Input candidate pixels: {result['original']:,}")
        print(f"Retained candidate pixels: {retained:,}")
        print(f"Excluded by occurrence threshold: {result['excluded']:,}")
        print(f"Candidates with invalid/missing water data: "
              f"{result['missing_water']:,}")
        print(f"Retained nominal area: {area_km2:.2f} km²")
        print(f"Output: {output_paths[threshold]}")

    print(
        "\nThese are water-occurrence screening scenarios, "
        "not confirmed flood extents."
    )


if __name__ == "__main__":
    main()
