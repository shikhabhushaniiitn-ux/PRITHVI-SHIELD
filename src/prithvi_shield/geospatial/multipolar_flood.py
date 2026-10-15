
"""Windowed VV + VH Sentinel-1 backscatter-change screening.

Output values:
    0   = valid pixel, not a candidate
    1   = candidate where both VV and VH decrease by at least
          the configured threshold
    255 = NoData or rejected input pixel
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


BASE = Path("data/processed/change_detection")

THRESHOLD_DB = -2.5
WINDOW = 512
NODATA_OUT = 255

OUTPUT = BASE / "candidate_both_pol_raw.tif"


def main():
    paths = {
        "pre_vv": BASE / "pre_vv_aoi_aligned.tif",
        "post_vv": BASE / "post_vv_aoi_aligned.tif",
        "pre_vh": BASE / "pre_vh_aoi_aligned.tif",
        "post_vh": BASE / "post_vh_aoi_aligned.tif",
    }

    # Check that every required input exists.
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing input raster: {path}")

    # Open all input rasters and validate their grids.
    with (
        rasterio.open(paths["pre_vv"]) as pvv,
        rasterio.open(paths["post_vv"]) as qvv,
        rasterio.open(paths["pre_vh"]) as pvh,
        rasterio.open(paths["post_vh"]) as qvh,
    ):
        datasets = (pvv, qvv, pvh, qvh)
        ref = pvv

        for name, ds in zip(
            ("post_vv", "pre_vh", "post_vh"),
            datasets[1:],
        ):
            if (
                ds.crs != ref.crs
                or ds.width != ref.width
                or ds.height != ref.height
                or not ds.transform.almost_equals(ref.transform)
            ):
                raise ValueError(
                    f"Input raster '{name}' is not aligned with pre_vv."
                )

        profile = ref.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=NODATA_OUT,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

        total_pixels = ref.width * ref.height
        valid_count = 0
        candidate_count = 0
        rejected_count = 0

        print("Starting VV + VH change screening...")
        print(f"Grid: {ref.width:,} x {ref.height:,}")
        print(f"Threshold: {THRESHOLD_DB} dB")
        print(f"Output: {OUTPUT}")

        # Write a new output raster, processing one window at a time.
        with rasterio.open(OUTPUT, "w", **profile) as dst:
            for row in range(0, ref.height, WINDOW):
                height = min(WINDOW, ref.height - row)

                for col in range(0, ref.width, WINDOW):
                    width = min(WINDOW, ref.width - col)
                    window = Window(col, row, width, height)

                    # Masked reads convert declared NoData pixels to NaN.
                    arrays = [
                        ds.read(1, window=window, masked=True)
                        .filled(np.nan)
                        for ds in datasets
                    ]

                    a, b, c, d = [
                        np.asarray(arr, dtype=np.float32)
                        for arr in arrays
                    ]

                    # All four backscatter values must be finite and positive.
                    valid = (
                        np.isfinite(a)
                        & np.isfinite(b)
                        & np.isfinite(c)
                        & np.isfinite(d)
                        & (a > 0)
                        & (b > 0)
                        & (c > 0)
                        & (d > 0)
                    )

                    vv_change = np.full(a.shape, np.nan, dtype=np.float32)
                    vh_change = np.full(a.shape, np.nan, dtype=np.float32)

                    # Compute dB change only at valid pixels.
                    with np.errstate(divide="ignore", invalid="ignore"):
                        vv_change[valid] = (
                            10.0 * np.log10(b[valid] / a[valid])
                        )
                        vh_change[valid] = (
                            10.0 * np.log10(d[valid] / c[valid])
                        )

                    # Both polarizations must meet the threshold.
                    candidate = (
                        valid
                        & (vv_change <= THRESHOLD_DB)
                        & (vh_change <= THRESHOLD_DB)
                    )

                    result = np.full(
                        a.shape, NODATA_OUT, dtype=np.uint8
                    )
                    result[valid] = 0
                    result[candidate] = 1

                    dst.write(result, 1, window=window)

                    valid_count += int(valid.sum())
                    candidate_count += int(candidate.sum())
                    rejected_count += int((~valid).sum())

        # Report counts from the processing calculations.
        print("\nScreening complete.")
        print(f"Total grid pixels: {total_pixels:,}")
        print(f"Valid pixels: {valid_count:,}")
        print(f"Candidate pixels: {candidate_count:,}")
        print(f"Rejected / NoData pixels: {rejected_count:,}")
        print(
            "Count check:",
            "PASS" if valid_count + rejected_count == total_pixels
            else "FAIL",
        )

        candidate_fraction = (
            candidate_count / valid_count if valid_count else 0.0
        )
        pixel_area_km2 = (
            abs(ref.transform.a * ref.transform.e
                - ref.transform.b * ref.transform.d)
            / 1_000_000
        )
        candidate_area_km2 = candidate_count * pixel_area_km2

        print(f"Candidate fraction of valid pixels: {candidate_fraction:.4%}")
        print(f"Nominal candidate pixel area: {candidate_area_km2:.3f} km²")
        print(f"Output raster: {OUTPUT}")
        print("Output encoding: 0=non-candidate, 1=candidate, 255=NoData")
        print(
            "Important: candidate area is not confirmed flooded area."
        )

    # Independently reopen the written raster and verify its pixel counts.
    written_counts = {0: 0, 1: 0, NODATA_OUT: 0}

    with rasterio.open(OUTPUT) as check:
        for _, window in check.block_windows(1):
            block = check.read(1, window=window)
            values, frequencies = np.unique(block, return_counts=True)

            for value, frequency in zip(values, frequencies):
                value = int(value)
                written_counts[value] = (
                    written_counts.get(value, 0) + int(frequency)
                )

    written_total = sum(written_counts.values())

    print("\nIndependent output-raster verification")
    print(f"Value 0 (valid non-candidate): {written_counts[0]:,}")
    print(f"Value 1 (candidate): {written_counts[1]:,}")
    print(f"Value 255 (NoData): {written_counts[255]:,}")
    print(f"Total output pixels: {written_total:,}")

    if written_total != total_pixels:
        raise RuntimeError("Output pixel count does not match the input grid.")

    if (
        written_counts[0] != valid_count - candidate_count
        or written_counts[1] != candidate_count
        or written_counts[255] != rejected_count
    ):
        raise RuntimeError(
            "Output counts do not match processing counts. "
            "Investigate the output raster before proceeding."
        )

    print("Output verification: PASS")


if __name__ == "__main__":
    main()
