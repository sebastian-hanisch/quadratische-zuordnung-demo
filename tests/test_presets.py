"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink verwirft ungültige Werte."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import qmh_constants as C
import qmh_evaluation as ev
import qmh_presets as PR

APP = str(Path(__file__).resolve().parent.parent / "app.py")
DEFAULT = "Standardfall: acht Bauteile"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(PR.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_choices():
    for p in C.PRESETS.values():
        assert p["n"] in C.SIZES and p["a"] in C.COMPONENT_CHOICES and p["b"] in C.COMPONENT_CHOICES and 0 <= p["seed"] <= C.SEED_MAX and 1 <= p["start"] <= C.START_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[DEFAULT]) == ev.Settings() and next(iter(C.PRESETS)) == DEFAULT


def test_setting_specs_cover_every_preset_key():
    assert set(PR.PRESET_KEYS.values()) == set(PR.SETTING_SPECS)


def test_the_preset_values_differ_from_the_default_where_the_help_says_so():
    assert C.PRESETS["Kleinste Halle: vier Bauteile"]["n"] == 4 and C.PRESETS["Sechs Bauteile: alle 720 Anordnungen"]["n"] == 6 and C.PRESETS["Neun Bauteile: 362 880 Anordnungen"]["n"] == 9
    assert C.PRESETS["Anderer Start (Nr. 7)"]["start"] == 7 and (C.PRESETS["Tausch zweier Bauteile (3 und 6)"]["a"], C.PRESETS["Tausch zweier Bauteile (3 und 6)"]["b"]) == (3, 6)
    import math
    assert math.factorial(6) == 720 and math.factorial(9) == 362880 and math.factorial(8) == 40320 and math.factorial(4) == 24


def test_permalink_takes_valid_values_and_drops_invalid_ones():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["seed"] = "7"
    at.query_params["n"] = "6"
    at.query_params["a"] = "3"
    at.query_params["b"] = "5"
    at.query_params["start"] = "12"
    at.run()
    assert not at.exception
    assert at.sidebar.number_input(key="seed_input").value == 7 and at.session_state["n_select"] == 6 and at.session_state["a_select"] == 3 and at.session_state["b_select"] == 5 and at.session_state["start_input"] == 12
    bad = AppTest.from_file(APP, default_timeout=300)
    bad.query_params["n"] = "7"
    bad.query_params["a"] = "11"
    bad.query_params["seed"] = "abc"
    bad.query_params["start"] = "0"
    bad.run()
    assert not bad.exception
    assert bad.session_state["n_select"] == C.DEFAULT_N and bad.session_state["a_select"] == C.DEFAULT_A and bad.session_state["seed_input"] == C.DEFAULT_SEED and bad.session_state["start_input"] == 1


def test_the_seed_is_clamped_to_its_bounds():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["seed"] = "99999999"
    at.run()
    assert at.session_state["seed_input"] == C.SEED_MAX
