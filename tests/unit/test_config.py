
from pathlib import Path

import yaml


def test_default_configuration_is_valid():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "default.yaml"
    )

    assert config_path.exists()

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    assert config["project"]["name"] == "PRITHVI-SHIELD"
    assert config["scenario"]["default_hazard"] == "flood"
    assert config["data"]["record_provenance"] is True