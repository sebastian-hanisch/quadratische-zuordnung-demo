"""Gemeinsame Fixtures: das Handbeispiel und eine kleine Instanz, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qmh_landscape as L  # noqa: E402
from qmh_instance import hand_instance, make_instance  # noqa: E402


@pytest.fixture
def hand():
    """Raster 2×2, Fluss A–B: 3, A–C: 1, B–D: 2, C–D: 4; Plätze 0 1 / 2 3 (Nachbarn Abstand 1, Diagonale 2). Handrechnung in der README."""
    return hand_instance()


@pytest.fixture(scope="session")
def land6():
    inst = make_instance(6, 35)
    return inst, L.landscape(inst)


@pytest.fixture(scope="session")
def land8():
    inst = make_instance(8, 35)
    return inst, L.landscape(inst)
