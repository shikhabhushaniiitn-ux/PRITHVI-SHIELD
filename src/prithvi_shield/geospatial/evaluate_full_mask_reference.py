"""Evaluate 50% and 80% historical-water candidate masks.

Reference date: 27 August 2026
Post-event Sentinel-1 date: 5 September 2026

Metrics measure agreement with the available reference, not definitive
event-time flood-detection accuracy.

Input rasters and reference files are read-only.
Existing 50% reports are retained at their original paths.
"""

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, transform as window_transform
from rasterio.warp import transform_geom


ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data/processed/change_detection"
REFERENCE = ROOT / "data/raw/reference/observed_flood_extent_20260827.geojson"

MASKS = [
    {
        "label": "50%",
        "description": "combined screened mask, historical water threshold 50%",
        "path": DATA_DIR / "candidate_combined_screened_50pct.tif",
        "csv": DATA_DIR / "full_mask_reference_metrics.csv",
        "txt": DATA_DIR / "full_mask_reference_metrics.txt",
    },
    {
        "label": "80%",
        "description": "combined screened mask, historical water threshold 80%",
        "path": DATA_DIR / "candidate_combined_screened_80pct.tif",
        "csv": DATA_DIR / "full_mask_reference_metrics_80pct.csv",
        "txt": DATA_DIR / "full_mask_reference_metrics_80pct.txt",
    },
]

COMPARISON_CSV = DATA_DIR / "full_mask_reference_comparison.csv"

BLOCK_SIZE = 512


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def evaluate_mask(mask_info, reference_geometries):
    """Evaluate one binary candidate mask in memory-efficient windows."""

    mask_path = mask_info["path"]

    if not mask_path.exists():
        raise FileNotFoundError(f"Candidate mask not found: {mask_path}")

    counts = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,
        "valid_pixels": 0,
        "nodata_pixels": 0,
    }

    with rasterio.open(mask_path) as src:
        if src.crs is None:
            raise ValueError(f"Candidate mask has no CRS: {mask_path}")

        if src.nodata is None:
            raise ValueError(
                f"Mask has no declared NoData value: {mask_path}"
            )

        print(f"\nEvaluating {mask_path.name}")
        print(f"CRS: {src.crs}")
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"NoData value: {src.nodata}")

        projected_geometries = [
            transform_geom("EPSG:4326", src.crs, geom)
            for geom in reference_geometries
        ]

        for row_off in range(0, src.height, BLOCK_SIZE):
            for col_off in range(0, src.width, BLOCK_SIZE):
                height = min(BLOCK_SIZE, src.height - row_off)
                width = min(BLOCK_SIZE, src.width - col_off)

                window = Window(
                    col_off, row_off, width, height
                )

                candidate = src.read(1, window=window)

                valid = candidate != src.nodata

                unexpected = valid & ~np.isin(candidate, [0, 1])
                if np.any(unexpected):
                    values = np.unique(candidate[unexpected])
                    raise ValueError(
                        f"Unexpected valid mask values in {mask_path.name}: "
                        f"{values[:10]}"
                    )

                reference = rasterize(
                    [(geom, 1) for geom in projected_geometries],
                    out_shape=(height, width),
                    transform=window_transform(window, src.transform),
                    fill=0,
                    all_touched=False,
                    dtype="uint8",
                ).astype(bool)

                candidate_positive = candidate == 1

                counts["TP"] += int(
                    np.count_nonzero(valid & candidate_positive & reference)
                )
                counts["FP"] += int(
                    np.count_nonzero(valid & candidate_positive & ~reference)
                )
                counts["FN"] += int(
                    np.count_nonzero(valid & ~candidate_positive & reference)
                )
                counts["TN"] += int(
                    np.count_nonzero(valid & ~candidate_positive & ~reference)
                )
                counts["valid_pixels"] += int(np.count_nonzero(valid))
                counts["nodata_pixels"] += int(np.count_nonzero(~valid))

        metadata = {
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "transform": tuple(src.transform),
            "nodata": src.nodata,
        }

    tp, fp = counts["TP"], counts["FP"]
    fn, tn = counts["FN"], counts["TN"]

    metrics = {
        **counts,
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "specificity": safe_ratio(tn, tn + fp),
        "F1": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "IoU": safe_ratio(tp, tp + fp + fn),
        "accuracy": safe_ratio(tp + tn, tp + fp + fn + tn),
    }

    return metadata, metrics


def write_reports(mask_info, metrics):
    """Write one mask's CSV and text report."""

    csv_path = mask_info["csv"]
    txt_path = mask_info["txt"]

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for key, value in metrics.items():
            writer.writerow([key, value])

    lines = [
        "PRITHVI-SHIELD FULL-MASK REFERENCE EVALUATION",
        "=" * 52,
        f"Candidate: {mask_info['description']}",
        "Reference: observed flood extent dated 27 August 2026",
        "Radar post-scene: 5 September 2026",
        "",
        f"True positives (TP): {metrics['TP']:,}",
        f"False positives (FP): {metrics['FP']:,}",
        f"False negatives (FN): {metrics['FN']:,}",
        f"True negatives (TN): {metrics['TN']:,}",
        f"Valid pixels: {metrics['valid_pixels']:,}",
        f"NoData pixels excluded: {metrics['nodata_pixels']:,}",
        "",
        f"Precision:   {metrics['precision']:.6f}",
        f"Recall:      {metrics['recall']:.6f}",
        f"Specificity: {metrics['specificity']:.6f}",
        f"F1 score:    {metrics['F1']:.6f}",
        f"IoU:         {metrics['IoU']:.6f}",
        f"Accuracy:    {metrics['accuracy']:.6f}",
        "",
        "INTERPRETATION LIMITATIONS",
        "- Reference absence is treated as a negative label within valid pixels.",
        "- Incomplete reference coverage can make apparent false positives misleading.",
        "- The reference and radar acquisitions have different dates.",
        "- Metrics measure agreement with this reference, not definitive flood accuracy.",
        "- Pixel-center rasterization is used (all_touched=False).",
        "- NoData pixels are excluded from all confusion-matrix metrics.",
    ]

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nCSV saved: {csv_path}")
    print(f"Text report saved: {txt_path}")


def main():
    if not REFERENCE.exists():
        raise FileNotFoundError(f"Reference GeoJSON not found: {REFERENCE}")

    with REFERENCE.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    geometries = [
        feature["geometry"]
        for feature in geojson.get("features", [])
        if feature.get("geometry")
    ]

    if not geometries:
        raise ValueError("No usable geometries found in reference GeoJSON.")

    results = []
    reference_metadata = None

    for mask_info in MASKS:
        metadata, metrics = evaluate_mask(mask_info, geometries)

        if reference_metadata is None:
            reference_metadata = metadata
        else:
            for key in ("crs", "width", "height", "transform"):
                if metadata[key] != reference_metadata[key]:
                    raise ValueError(
                        f"Mask grid mismatch for {key}: "
                        f"{mask_info['path'].name}. "
                        "Comparison stopped to avoid unfair metrics."
                    )

        results.append((mask_info, metrics))
        write_reports(mask_info, metrics)

    # Save a side-by-side comparison without replacing either individual report.
    comparison_fields = [
        "threshold",
        "TP",
        "FP",
        "FN",
        "TN",
        "valid_pixels",
        "nodata_pixels",
        "precision",
        "recall",
        "specificity",
        "F1",
        "IoU",
        "accuracy",
    ]

    with COMPARISON_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=comparison_fields)
        writer.writeheader()

        for mask_info, metrics in results:
            writer.writerow({
                "threshold": mask_info["label"],
                **metrics,
            })

    print("\nTHRESHOLD COMPARISON")
    print("=" * 52)

    for mask_info, metrics in results:
        print(
            f"{mask_info['label']:>3} | "
            f"Precision={metrics['precision']:.6f} | "
            f"Recall={metrics['recall']:.6f} | "
            f"F1={metrics['F1']:.6f} | "
            f"IoU={metrics['IoU']:.6f} | "
            f"Valid={metrics['valid_pixels']:,}"
        )

    print(f"\nComparison CSV saved: {COMPARISON_CSV}")
    print("\nEvaluation completed. No input raster or reference was modified.")


if __name__ == "__main__":
    main()