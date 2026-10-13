
"""Geometry-based coverage assessment for satellite scenes.

Coverage fractions are measured against a provisional WGS84 AOI rectangle.
They are not a substitute for a validated study-area polygon or image QA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pyproj import CRS, Transformer
from shapely import transform
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry


WGS84_CRS = "EPSG:4326"
DEFAULT_PROJECTED_CRS = "EPSG:32645"


@dataclass(frozen=True)
class SceneCoverageAssessment:
    """Coverage of a provisional AOI by one satellite scene."""

    intersects_aoi: bool
    aoi_area_km2: float
    intersection_area_km2: float
    coverage_fraction: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "intersects_aoi": self.intersects_aoi,
            "aoi_area_km2": round(self.aoi_area_km2, 6),
            "intersection_area_km2": round(
                self.intersection_area_km2, 6
            ),
            "coverage_fraction": round(self.coverage_fraction, 6),
            "coverage_percent": round(
                self.coverage_fraction * 100, 2
            ),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SharedCoverageAssessment:
    """Shared AOI coverage by a pre-event and post-event scene."""

    shared_area_km2: float
    shared_coverage_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "shared_area_km2": round(self.shared_area_km2, 6),
            "shared_coverage_fraction": round(
                self.shared_coverage_fraction, 6
            ),
            "shared_coverage_percent": round(
                self.shared_coverage_fraction * 100, 2
            ),
        }


def _validate_aoi_bbox(
    aoi_bbox: tuple[float, float, float, float],
) -> None:
    """Validate a WGS84 bbox ordered west, south, east, north."""
    if len(aoi_bbox) != 4:
        raise ValueError(
            "aoi_bbox must be (west, south, east, north)."
        )

    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in aoi_bbox
    ):
        raise ValueError("AOI bbox values must be numeric.")

    if not all(math.isfinite(value) for value in aoi_bbox):
        raise ValueError("AOI bbox values must be finite.")

    west, south, east, north = aoi_bbox

    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError("AOI longitudes must be within [-180, 180].")

    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValueError("AOI latitudes must be within [-90, 90].")

    if west >= east or south >= north:
        raise ValueError("AOI bbox must have positive area.")


def _parse_polygonal_geometry(
    geometry: dict[str, Any] | BaseGeometry,
    name: str,
) -> BaseGeometry:
    """Parse and validate a Polygon or MultiPolygon geometry."""
    try:
        parsed = (
            shape(geometry)
            if isinstance(geometry, dict)
            else geometry
        )
    except Exception as exc:
        raise ValueError(
            f"{name} is not valid GeoJSON geometry."
        ) from exc

    if not isinstance(parsed, BaseGeometry):
        raise ValueError(
            f"{name} must be a Shapely or GeoJSON geometry."
        )

    if parsed.is_empty:
        raise ValueError(f"{name} must not be empty.")

    if parsed.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(
            f"{name} must be a Polygon or MultiPolygon; "
            f"received {parsed.geom_type}."
        )

    if not parsed.is_valid:
        raise ValueError(f"{name} is geometrically invalid.")

    return parsed


def _project_geometry(
    geometry: BaseGeometry,
    projected_crs: str,
) -> BaseGeometry:
    """Project a WGS84 geometry into a projected coordinate system."""
    try:
        target_crs = CRS.from_user_input(projected_crs)
    except Exception as exc:
        raise ValueError("projected_crs is not a valid CRS.") from exc

    if not target_crs.is_projected:
        raise ValueError(
            "projected_crs must be a projected coordinate "
            "reference system."
        )

    transformer = Transformer.from_crs(
        WGS84_CRS,
        target_crs,
        always_xy=True,
    )

    projected = transform(
        geometry,
        transformer.transform,
        interleaved=False,
    )

    if projected.is_empty or not projected.is_valid:
        raise ValueError(
            "Geometry became empty or invalid after projection."
        )

    return projected


def assess_scene_coverage(
    footprint_geometry: dict[str, Any] | BaseGeometry,
    aoi_bbox: tuple[float, float, float, float],
    *,
    projected_crs: str = DEFAULT_PROJECTED_CRS,
) -> SceneCoverageAssessment:
    """Calculate the fraction of the provisional AOI covered by a scene.

    Both input geometries are assumed to use WGS84 (EPSG:4326).
    Coverage fraction = intersection area / total AOI area.
    """
    _validate_aoi_bbox(aoi_bbox)

    footprint = _parse_polygonal_geometry(
        footprint_geometry,
        "footprint_geometry",
    )

    aoi_wgs84 = box(*aoi_bbox)

    projected_aoi = _project_geometry(aoi_wgs84, projected_crs)
    projected_footprint = _project_geometry(
        footprint,
        projected_crs,
    )

    aoi_area = projected_aoi.area
    if aoi_area <= 0:
        raise ValueError("Projected AOI must have positive area.")

    intersection = projected_aoi.intersection(projected_footprint)
    intersection_area = (
        intersection.area if not intersection.is_empty else 0.0
    )

    # Guard against tiny floating-point differences at polygon boundaries.
    intersection_area = min(max(intersection_area, 0.0), aoi_area)
    fraction = intersection_area / aoi_area

    if fraction <= 0:
        reasons = ("Scene footprint does not intersect the AOI.",)
    elif fraction < 0.999999:
        reasons = (
            "Scene intersects the AOI but does not cover the entire "
            "rectangle.",
        )
    else:
        reasons = (
            "Scene footprint covers the full provisional AOI rectangle.",
        )

    return SceneCoverageAssessment(
        intersects_aoi=intersection_area > 0,
        aoi_area_km2=aoi_area / 1_000_000,
        intersection_area_km2=intersection_area / 1_000_000,
        coverage_fraction=fraction,
        reasons=reasons,
    )


def assess_shared_coverage(
    first_footprint_geometry: dict[str, Any] | BaseGeometry,
    second_footprint_geometry: dict[str, Any] | BaseGeometry,
    aoi_bbox: tuple[float, float, float, float],
    *,
    projected_crs: str = DEFAULT_PROJECTED_CRS,
) -> SharedCoverageAssessment:
    """Calculate the AOI fraction covered by both scene footprints."""
    _validate_aoi_bbox(aoi_bbox)

    first = _parse_polygonal_geometry(
        first_footprint_geometry,
        "first_footprint_geometry",
    )
    second = _parse_polygonal_geometry(
        second_footprint_geometry,
        "second_footprint_geometry",
    )

    aoi_projected = _project_geometry(
        box(*aoi_bbox),
        projected_crs,
    )
    first_projected = _project_geometry(first, projected_crs)
    second_projected = _project_geometry(second, projected_crs)

    aoi_area = aoi_projected.area
    if aoi_area <= 0:
        raise ValueError("Projected AOI must have positive area.")

    first_clipped = aoi_projected.intersection(first_projected)
    second_clipped = aoi_projected.intersection(second_projected)

    if first_clipped.is_empty or second_clipped.is_empty:
        shared_area = 0.0
    else:
        shared_area = first_clipped.intersection(second_clipped).area

    shared_area = min(max(shared_area, 0.0), aoi_area)

    return SharedCoverageAssessment(
        shared_area_km2=shared_area / 1_000_000,
        shared_coverage_fraction=shared_area / aoi_area,
    )