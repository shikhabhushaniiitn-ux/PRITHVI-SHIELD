
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse


VALID_POLARIZATIONS = {"VV", "VH", "HH", "HV"}
VALID_ORBIT_DIRECTIONS = {"ASCENDING", "DESCENDING"}


@dataclass(frozen=True)
class SatelliteObservation:
    """Metadata describing a satellite observation."""

    product_id: str
    collection_id: str
    acquired_at: datetime
    platform: str
    product_type: str
    polarizations: tuple[str, ...]
    catalog_url: str
    source_url: Optional[str] = None
    orbit_direction: Optional[str] = None

    def __post_init__(self) -> None:
        # Required text fields must not be empty.
        required_fields = {
            "product_id": self.product_id,
            "collection_id": self.collection_id,
            "platform": self.platform,
            "product_type": self.product_type,
            "catalog_url": self.catalog_url,
        }

        for field_name, value in required_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string."
                )

        # Acquisition timestamps must include a timezone.
        if (
            not isinstance(self.acquired_at, datetime)
            or self.acquired_at.tzinfo is None
            or self.acquired_at.utcoffset() is None
        ):
            raise ValueError(
                "acquired_at must be a timezone-aware datetime."
            )

        # Validate polarization metadata.
        if not self.polarizations:
            raise ValueError(
                "At least one polarization is required."
            )

        normalized_polarizations = tuple(
            polarization.upper()
            for polarization in self.polarizations
        )

        if not set(normalized_polarizations).issubset(
            VALID_POLARIZATIONS
        ):
            raise ValueError(
                "Polarizations must be VV, VH, HH, or HV."
            )

        object.__setattr__(
            self,
            "polarizations",
            normalized_polarizations,
        )

        # Validate optional orbit direction.
        if self.orbit_direction is not None:
            direction = self.orbit_direction.upper()

            if direction not in VALID_ORBIT_DIRECTIONS:
                raise ValueError(
                    "orbit_direction must be ASCENDING "
                    "or DESCENDING."
                )

            object.__setattr__(self, "orbit_direction", direction)

        # Catalogue and source references must be HTTP(S) URLs.
        self._validate_url("catalog_url", self.catalog_url)

        if self.source_url is not None:
            self._validate_url("source_url", self.source_url)

    @staticmethod
    def _validate_url(field_name: str, value: str) -> None:
        parsed = urlparse(value)

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
        ):
            raise ValueError(
                f"{field_name} must be a valid HTTP(S) URL."
            )

    def to_dict(self) -> dict:
        """Return JSON-friendly metadata for provenance records."""
        result = asdict(self)
        result["acquired_at"] = self.acquired_at.isoformat()
        result["polarizations"] = list(self.polarizations)
        return result
    