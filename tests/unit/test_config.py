
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as file:
        return yaml.safe_load(file)


def test_default_configuration_is_valid():
    config_path = PROJECT_ROOT / "configs" / "default.yaml"

    assert config_path.exists()

    config = load_yaml(config_path)

    assert config["project"]["name"] == "PRITHVI-SHIELD"
    assert config["scenario"]["default_hazard"] == "flood"
    assert config["data"]["record_provenance"] is True


def test_default_scenario_configuration_is_valid():
    config_path = (
        PROJECT_ROOT
        / "configs"
        / "scenarios"
        / "default.yaml"
    )

    assert config_path.exists()

    config = load_yaml(config_path)

    assert config["project"]["name"] == "PRITHVI-SHIELD"
    assert (
        config["scenario"]["default_region"]
        == "trishuli_river_corridor"
    )
    assert config["scenario"]["default_hazard"] == "flood"
    assert config["data"]["primary_source"] == "sentinel-1-grd"
    assert config["data"]["record_provenance"] is True
    assert (
        config["data"]["reference_maps_for_validation_only"]
        is True
    )