"""Load and validate scenario files from scenarios/."""

from __future__ import annotations

import json
from pathlib import Path

from .config import SCENARIOS_DIR
from .models import Scenario, ScenarioStub

AnyScenario = Scenario | ScenarioStub


def load_scenario_file(path: Path) -> AnyScenario:
    """Parse one file. A file with a 'problem' field and no 'opening_line' is a stub."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    if "problem" in data and "opening_line" not in data:
        scenario: AnyScenario = ScenarioStub.model_validate(data)
    else:
        scenario = Scenario.model_validate(data)
    if scenario.id != path.stem:
        raise ValueError(f"{path.name}: id {scenario.id!r} does not match the file name")
    return scenario


def load_all(directory: Path = SCENARIOS_DIR) -> list[AnyScenario]:
    return [load_scenario_file(p) for p in sorted(directory.glob("*.json"))]


def load_full(directory: Path = SCENARIOS_DIR) -> list[Scenario]:
    return [s for s in load_all(directory) if isinstance(s, Scenario)]


def get_scenario(scenario_id: str, directory: Path = SCENARIOS_DIR) -> AnyScenario:
    path = directory / f"{scenario_id}.json"
    if not path.exists():
        known = ", ".join(p.stem for p in sorted(directory.glob("*.json")))
        raise FileNotFoundError(f"no scenario {scenario_id!r}. Known: {known}")
    return load_scenario_file(path)
