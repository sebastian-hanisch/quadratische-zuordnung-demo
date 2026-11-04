"""Randfälle (DEMO-PLAYBOOK §1d): kleinste Größen, Fluss null, vollständiger Fluss, Zwillinge, begrenzte Regler, Eingaben außerhalb des Bereichs."""

import numpy as np
import pytest

import qmh_cost as K
import qmh_evaluation as ev
import qmh_instance as I
import qmh_landscape as L


def _inst(F, rows, cols):
    n = len(F)
    return I.Instance(n, rows, cols, np.array(F, dtype=np.int16), I.grid_distance(rows, cols), I.component_names(n), (), 0)


def test_two_and_three_components():
    two = L.landscape(_inst([[0, 5], [5, 0]], 1, 2))
    assert two.perms == 2 and two.opt == 5 and two.n_opt == 2 and two.success == 1.0 and two.aut == 2
    three = L.landscape(_inst([[0, 2, 1], [2, 0, 0], [1, 0, 0]], 1, 3))
    assert three.perms == 6 and three.opt == 2 * 1 + 1 * 1 and three.n_opt == 2                      # A in der Mitte, B und C außen (zwei Spiegelbilder)


def test_zero_flow_makes_every_arrangement_optimal():
    ls = L.landscape(_inst(np.zeros((4, 4), dtype=int), 2, 2))
    assert ls.opt == 0 and ls.n_opt == 24 and ls.n_local == 24 and ls.n_local_strict == 0 and ls.success == 1.0 and ls.mean_steps == 0.0 and ls.max_steps == 0


def test_a_complete_flow_makes_every_arrangement_equal():
    F = np.ones((4, 4), dtype=int) - np.eye(4, dtype=int)
    ls = L.landscape(_inst(F, 2, 2))
    assert ls.n_opt == 24 and ls.cost_values.tolist() == [ls.opt] and ls.n_local_strict == 0


def test_twins_give_equal_costs_for_every_arrangement_when_swapped():
    """Zwei Bauteile mit gleichem Fluss zu allen anderen sind Zwillinge: ihr Tausch ändert die Kosten nie (Δ = 0), die Zahl der Optima ist durch zwei teilbar."""
    F = np.zeros((4, 4), dtype=int)
    F[0, 2] = F[2, 0] = F[1, 2] = F[2, 1] = 3                                            # Bauteil 0 und 1 sind Zwillinge (beide nur mit 2 verbunden)
    inst = _inst(F, 2, 2)
    from itertools import permutations
    for p in permutations(range(4)):
        assert K.swap_delta(list(p), 0, 1, inst.F, inst.D) == 0 and K.cost(K.swapped(p, 0, 1), inst.F, inst.D) == K.cost(p, inst.F, inst.D)
    ls = L.landscape(inst)
    assert ls.n_opt % 2 == 0 and ls.n_local_strict == 0                                  # Zwillinge machen jedes lokale Optimum schwach


def test_a_one_row_grid_has_two_automorphisms():
    ls = L.landscape(_inst([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]], 1, 5))
    assert ls.aut == 2 and ls.opt == 4 and ls.n_opt == 2                                  # eine Kette auf einer Zeile: die Kette selbst und ihr Spiegelbild


def test_the_pair_of_components_is_clamped_to_n_and_never_equal():
    for n in (4, 6, 8, 9):
        for a in range(1, 10):
            for b in range(1, 10):
                x, y = ev.pair_of(ev.Settings(n=n, a=a, b=b))
                assert 0 <= x < n and 0 <= y < n and x != y


def test_extreme_seeds_and_starts_work():
    for seed in (0, 999999):
        inst = I.make_instance(6, seed)
        assert L.landscape(inst).perms == 720
    s = ev.Settings(n=6, seed=0, start=999999)
    ls = ev.landscape_of(s)
    assert ev.descent_view(s, ls)["end_cost"] >= ls.opt


def test_inputs_out_of_range_are_rejected():
    with pytest.raises(ValueError):
        I.make_instance(5, 1)
    with pytest.raises(ValueError):
        K.pair_costs([0, 1], np.zeros((3, 3)), np.zeros((3, 3))) if False else K.cost([0, 1, 2, 2], np.zeros((4, 4)), np.zeros((4, 4)))


def test_a_single_instance_in_the_growth_table_has_equal_median_min_and_max():
    rows = ev.growth_rows(sizes=(4,), seeds=(7,))
    assert rows[0]["local_orbits"] == rows[0]["local_min"] == rows[0]["local_max"] and rows[0]["seeds"] == 1
