"""Jede Zahl aus README.md, den Hilfetexten der Presets und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Die Aufzählung ist deterministisch (SplitMix64, ganzzahlige Kosten): alle Zahlen unten sind **exakt**, nicht Bänder. Zeiten stehen nie in Tests."""

import math
from pathlib import Path

import numpy as np
import pytest

import qmh_constants as C
import qmh_evaluation as ev
import qmh_instance as I
import qmh_landscape as L


def test_the_hand_example_of_the_readme(hand):
    """README: Raster 2×2, Fluss A–B 3, A–C 1, B–D 2, C–D 4: Identität kostet 10; die 24 Anordnungen kosten 10, 13, 17 je achtmal; alle 8 Optima sind schwache, keines ist streng; Bahnen = 1."""
    hv = ev.hand_view()
    assert hv["values"] == {10: 8, 13: 8, 17: 8} and hv["rows"][0][0] == 10 and len(hv["rows"]) == 24
    ls = L.landscape(hand)
    assert (ls.opt, ls.n_opt, ls.aut, ls.optima_orbits, ls.n_local, ls.n_local_strict, ls.success) == (10, 8, 8, 1.0, 8, 0, 1.0)


def test_the_default_hall_with_eight_components():
    """README: n = 8, Seed 35: 40 320 Anordnungen, Optimum 31 (4 Anordnungen, 1 Bahn), schlechteste 69, Mittel 50,0; 112 schwache lokale Optima (28,0 je Bahn), davon 8 streng, 108 davon kein globales Optimum;
    Erfolg eines Abstiegs 26,6 % (10 728 von 40 320 Starts), im Mittel 2,97 Tausche je Abstieg, höchstens 6."""
    s = ev.Settings()
    ls = ev.landscape_of(s)
    assert (ls.perms, ls.opt, ls.n_opt, ls.aut, ls.optima_orbits, int(ls.cost_values[-1])) == (40320, 31, 4, 4, 1.0, 69) and ls.mean_cost == pytest.approx(50.0)
    assert (ls.n_local, ls.local_orbits, ls.n_local_strict) == (112, 28.0, 8) and int((ls.local_cost != ls.opt).sum()) == 108
    assert int(ls.local_basin[ls.local_cost == ls.opt].sum()) == 10728 and ls.success == pytest.approx(10728 / 40320) and ls.mean_steps == pytest.approx(2.9677, abs=1e-4) and ls.max_steps == 6
    assert ls.restarts == pytest.approx(3.7585, abs=1e-3)


def test_the_experiments_three_and_four_for_the_default_start():
    """README: Start Nr. 1 hat die Kosten 45; Tausch von IC1 und C1: Kosten 47 (Δ = +2, Formel = Neuberechnung); 7 der 28 Tausche verbessern, der beste (IC1, R1... Paar 1–3) hat Δ = −10;
    der Abstieg 45 → 35 → 32 → 31 endet nach 3 Tauschen im Optimum, sein Endpunkt hat ein Einzugsgebiet von 2 682 Starts; das lokale Optimum mit dem größten Einzugsgebiet, das nicht global ist, kostet 32 (817 Starts)."""
    s = ev.Settings()
    ls = ev.landscape_of(s)
    sv, dv = ev.swap_view(s, ls), ev.descent_view(s, ls)
    assert (sv["cost_p"], sv["cost_q"], sv["delta_formula"], sv["delta_recomputed"], sv["improving"], sv["is_local"]) == (45, 47, 2, 2, 7, False) and sv["best"] == (1, 3, -10)
    assert sv["example"]["cost"] == 32 and sv["example"]["basin"] == 817 and not sv["example"]["strict"]
    assert dv["costs"] == [45, 35, 32, 31] and dv["steps"] == 3 and dv["global"] and dv["basin"] == 2682


def test_six_and_four_components_of_the_readme():
    """README: n = 6, Seed 35: 720 Anordnungen, Optimum 26 (8 Anordnungen, 2 Bahnen), Mittel 33,3, schlechteste 41; 12 lokale Optima (3,0 je Bahn), 4 streng, 4 davon kein globales Optimum;
    Erfolg 81,1 % (584 von 720); n = 4, Seed 35: Optimum 26, schlechteste 27 (die Kosten nehmen nur zwei Werte an: 26 für 8, 27 für 16 Anordnungen)."""
    ls6 = L.landscape(I.make_instance(6, 35))
    assert (ls6.perms, ls6.opt, ls6.n_opt, ls6.optima_orbits, int(ls6.cost_values[-1])) == (720, 26, 8, 2.0, 41) and ls6.mean_cost == pytest.approx(100 / 3)
    assert (ls6.n_local, ls6.local_orbits, ls6.n_local_strict, int((ls6.local_cost != ls6.opt).sum())) == (12, 3.0, 4, 4) and int(ls6.local_basin[ls6.local_cost == ls6.opt].sum()) == 584
    ls4 = L.landscape(I.make_instance(4, 35))
    assert dict(zip(ls4.cost_values.tolist(), ls4.cost_counts.tolist())) == {26: 8, 27: 16} and ls4.success == 1.0


@pytest.fixture(scope="module")
def sweep():
    return {r["n"]: r for r in ev.growth_rows(extra={9: C.SWEEP_SEEDS_9})}


def test_the_growth_table_of_the_readme(sweep):
    """README: Median über feste Instanzen (20 für n = 4, 6, 8; 5 für n = 9): lokale Optima je Bahn 1,5 / 7,0 / 28,0 / 41,0 (Spannen 1,0 bis 3,0; 2,0 bis 16,0; 12,0 bis 63,0; 36,0 bis 66,0);
    Erfolg eines Abstiegs 100 % / 81,4 % / 35,9 % / 26,6 % (Spannen 28,9 bis 100 % bei n = 6, 11,6 bis 81,6 % bei n = 8, 17,6 bis 52,4 % bei n = 9)."""
    assert [sweep[n]["seeds"] for n in (4, 6, 8, 9)] == [20, 20, 20, 5] and [sweep[n]["perms"] for n in (4, 6, 8, 9)] == [24, 720, 40320, 362880]
    assert [sweep[n]["local_orbits"] for n in (4, 6, 8, 9)] == [1.5, 7.0, 28.0, 41.0]
    assert [(sweep[n]["local_min"], sweep[n]["local_max"]) for n in (4, 6, 8, 9)] == [(1.0, 3.0), (2.0, 16.0), (12.0, 63.0), (36.0, 66.0)]
    assert sweep[4]["success"] == 1.0 and sweep[6]["success"] == pytest.approx(0.81389, abs=1e-4) and sweep[8]["success"] == pytest.approx(0.35928, abs=1e-4) and sweep[9]["success"] == pytest.approx(0.2661, abs=1e-4)
    assert (sweep[6]["success_min"], sweep[6]["success_max"]) == (pytest.approx(0.28889, abs=1e-4), 1.0) and (sweep[8]["success_min"], sweep[8]["success_max"]) == (pytest.approx(0.11587, abs=1e-4), pytest.approx(0.81627, abs=1e-4))
    assert (sweep[9]["success_min"], sweep[9]["success_max"]) == (pytest.approx(0.1757, abs=1e-4), pytest.approx(0.5237, abs=1e-4))
    assert sweep[8]["steps"] == pytest.approx(3.0319, abs=1e-3) and sweep[6]["steps"] == pytest.approx(1.8308, abs=1e-3)


def test_the_growth_verdict_of_the_readme_holds_for_the_table(sweep):
    kind, text = ev.verdict_growth([sweep[n] for n in (4, 6, 8, 9)])
    assert kind == "success" and "24 auf 362 880" in text and "n = 8: 28 lokale Optima je Bahn, Erfolg eines Starts 36 % (20 Instanzen)" in text


def test_the_nine_component_preset_instance():
    """Preset „Neun Bauteile“: Seed 100000: Optimum 82 (8 Anordnungen, 1 Bahn), 288 lokale Optima (36,0 je Bahn), 40 streng, 280 davon kein globales Optimum, Erfolg 17,6 %."""
    ls = L.landscape(I.make_instance(9, 100000))
    assert (ls.perms, ls.opt, ls.n_opt, ls.aut, ls.n_local, ls.local_orbits, ls.n_local_strict, int((ls.local_cost != ls.opt).sum())) == (362880, 82, 8, 8, 288, 36.0, 40, 280)
    assert ls.success == pytest.approx(0.1757, abs=1e-4)


def test_the_counts_in_the_texts():
    """Texte: n! für n = 4, 6, 8, 9 (24, 720, 40 320, 362 880), 12! = 479 001 600, 15 Tausche bei n = 6, 28 bei n = 8; Rasterautomorphismen 2×2 und 3×3: 8, 2×3 und 2×4: 4."""
    assert [math.factorial(n) for n in C.SIZES] == [24, 720, 40320, 362880] and C.n_text(math.factorial(12)) == "479 001 600"
    assert [n * (n - 1) // 2 for n in (6, 8)] == [15, 28]
    assert [len(L.automorphisms(*C.GRID[n])) for n in C.SIZES] == [8, 4, 4, 8]


def test_the_rng_gives_the_same_stream_as_the_one_of_the_leiterplatte_demo():
    """README: `qmh_rng.py` ist `lpl_rng.py` der Leiterplatte-Demo (gleiche Zahlenfolge für jede Saat; sofern das Schwester-Repo nebenan liegt)."""
    import importlib.util

    sibling = Path(__file__).resolve().parent.parent.parent / "leiterplatte-demo" / "lpl_rng.py"
    if not sibling.exists():
        pytest.skip("Schwester-Repo liegt nicht nebenan")
    spec = importlib.util.spec_from_file_location("lpl_rng_sibling", sibling)
    other = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(other)
    from qmh_rng import SplitMix64
    for seed in (0, 1, 35, 2**64 - 1):
        a, b = SplitMix64(seed), other.SplitMix64(seed)
        assert [a.next() for _ in range(50)] == [b.next() for _ in range(50)] and a.below(7) == b.below(7)
