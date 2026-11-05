"""Every scenario file must validate, and none may give away its root cause."""

import json
import re
from typing import get_args

import pytest

from helpdesk.config import SCENARIOS_DIR
from helpdesk.models import Category, Scenario, ScenarioStub
from helpdesk.scenarios import load_all, load_full, load_scenario_file

FILES = sorted(SCENARIOS_DIR.glob("*.json"))
FULL = load_full()


def test_thirty_scenario_files():
    assert len(FILES) == 30


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_scenario_file_validates(path):
    scenario = load_scenario_file(path)
    assert isinstance(scenario, (Scenario, ScenarioStub))


def test_ten_full_and_twenty_stubs():
    scenarios = load_all()
    assert sum(isinstance(s, Scenario) for s in scenarios) == 10
    assert sum(isinstance(s, ScenarioStub) for s in scenarios) == 20


def test_stubs_have_only_id_tier_problem():
    for path in FILES:
        data = json.loads(path.read_text())
        if "opening_line" not in data:
            assert set(data) == {"id", "tier", "problem"}, path.name


def test_ids_are_unique_and_match_file_names():
    ids = [s.id for s in load_all()]
    assert len(ids) == len(set(ids))
    assert ids == [p.stem for p in FILES]


def test_file_names_are_neutral():
    # Names are sequential ids only, so they can't hint at the cause.
    for path in FILES:
        assert re.fullmatch(r"hd-\d{3}\.json", path.name), path.name


def test_full_scenarios_cover_every_category():
    assert {s.category for s in FULL} == set(get_args(Category))


def test_tiers_and_outcomes_are_mixed():
    assert {s.tier for s in FULL} == {1, 2, 3}
    assert {s.tier for s in load_all()} == {1, 2, 3}
    assert any(s.must_escalate for s in FULL)
    assert any(not s.must_escalate for s in FULL)


# Words that may legitimately appear in both an opening line and its root cause: they name the
# problem area or device, not the cause.
_AREA_WORDS = {
    "computer", "laptop", "desktop", "phone", "printer", "internet", "wifi", "network",
    "email", "emails", "office", "windows", "work",
}
_STOPWORDS = {
    "about", "after", "again", "also", "because", "been", "being", "cannot", "could", "does",
    "doesn", "done", "each", "even", "every", "from", "have", "having", "here", "into", "just",
    "keeps", "like", "more", "much", "never", "nothing", "only", "other", "over", "same", "says", "should",
    "since", "some", "still", "than", "that", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "very", "want", "were", "what", "when", "where", "which", "while",
    "will", "with", "would", "your",
}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if len(w) >= 4 and w not in _STOPWORDS and w not in _AREA_WORDS}


@pytest.mark.parametrize("scenario", FULL, ids=lambda s: s.id)
def test_opening_line_does_not_reveal_root_cause(scenario):
    shared = _content_words(scenario.opening_line) & _content_words(scenario.hidden_root_cause)
    assert not shared, f"{scenario.id} opening line shares cause words: {sorted(shared)}"


@pytest.mark.parametrize("scenario", FULL, ids=lambda s: s.id)
def test_opening_line_is_not_a_diagnosis(scenario):
    # A real caller describes symptoms; they don't name the fix steps.
    for step in scenario.correct_fix_steps:
        quoted = re.findall(r"'([^']+)'", step)
        for phrase in quoted:
            assert phrase.lower() not in scenario.opening_line.lower(), (scenario.id, phrase)
