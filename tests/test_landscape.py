"""Die Landschaft: Aufzählung gegen Einzelrechnung, lokale Optima, Abstieg, Einzugsgebiete, Bahnen (Eigenschaftstests)."""

from itertools import permutations

import numpy as np
import pytest

import qmh_cost as K
import qmh_instance as I
import qmh_landscape as L


def test_the_hand_example_landscape(hand):
    ls = L.landscape(hand)
    assert ls.perms == 24 and ls.opt == 10 and ls.n_opt == 8 and ls.aut == 8 and ls.optima_orbits == 1.0
    assert dict(zip(ls.cost_values.tolist(), ls.cost_counts.tolist())) == {10: 8, 13: 8, 17: 8}
    assert ls.n_local == 8 and ls.n_local_strict == 0 and ls.success == 1.0                  # alle lokalen Optima sind global; Gleichstände unter den Spiegelungen
    assert ls.mean_cost == pytest.approx(40 / 3)


def test_the_enumeration_agrees_with_the_single_cost(land6):
    inst, ls = land6
    brute = {p: K.cost(p, inst.F, inst.D) for p in permutations(range(6))}
    assert ls.opt == min(brute.values()) and ls.n_opt == sum(1 for v in brute.values() if v == ls.opt)
    assert ls.mean_cost == pytest.approx(np.mean(list(brute.values())))
    assert sorted(map(tuple, ls.optima.tolist())) == sorted(p for p, v in brute.items() if v == ls.opt)
    assert int(ls.cost_counts.sum()) == 720 and ls.cost_values[0] == ls.opt


def test_the_local_optima_are_exactly_the_arrangements_without_an_improving_swap(land6):
    inst, ls = land6
    pairs = K.pair_list(6)
    weak, strict = set(), set()
    for p in permutations(range(6)):
        d = [K.swap_delta(p, a, b, inst.F, inst.D) for a, b in pairs]
        if min(d) >= 0:
            weak.add(p)
        if min(d) > 0:
            strict.add(p)
    assert set(map(tuple, ls.local.tolist())) == weak and {tuple(x) for x, s in zip(ls.local.tolist(), ls.local_strict) if s} == strict


def test_every_global_optimum_is_a_local_optimum_and_the_basins_cover_all_starts(land8):
    inst, ls = land8
    local = set(map(tuple, ls.local.tolist()))
    assert all(tuple(o) in local for o in ls.optima.tolist())
    assert int(ls.local_basin.sum()) == ls.perms == 40320 and (ls.local_basin >= 1).all()          # jeder Start endet in genau einem lokalen Optimum, jedes ist sein eigener Endpunkt
    assert ls.success == pytest.approx(ls.local_basin[ls.local_cost == ls.opt].sum() / ls.perms)


def test_the_scalar_descent_agrees_with_the_vectorised_one(land6):
    inst, ls = land6
    ends = {}
    for p in permutations(range(6)):
        path = L.descent(inst, list(p))
        costs = [K.cost(x, inst.F, inst.D) for x in path]
        assert all(costs[i + 1] < costs[i] for i in range(len(costs) - 1))                          # jeder Schritt senkt die Kosten
        ends[tuple(path[-1])] = ends.get(tuple(path[-1]), 0) + 1
    assert {tuple(x): int(b) for x, b in zip(ls.local.tolist(), ls.local_basin)} == ends


def test_basin_of_and_the_descent_from_a_local_optimum(land6):
    inst, ls = land6
    i = int(np.argmax(ls.local_basin))
    assert L.basin_of(ls, ls.local[i]) == int(ls.local_basin[i]) and L.descent(inst, ls.local[i].tolist()) == [ls.local[i].tolist()]
    assert L.basin_of(ls, [0, 0, 0, 0, 0, 0]) == 0                                                    # keine Anordnung, kein Einzugsgebiet


def test_automorphism_counts_of_the_grids():
    assert [len(L.automorphisms(*g)) for g in ((2, 2), (2, 3), (2, 4), (3, 3), (2, 5), (3, 4), (1, 5))] == [8, 4, 4, 8, 4, 4, 2]


def test_the_number_of_optima_is_a_multiple_of_the_automorphisms_when_no_component_is_a_twin(land8):
    inst, ls = land8
    assert ls.n_opt % ls.aut == 0 and ls.n_local % ls.aut == 0


def test_the_expected_restarts_are_the_inverse_of_the_success(land8):
    _, ls = land8
    assert ls.restarts == pytest.approx(1.0 / ls.success) and 0 < ls.success <= 1 and ls.mean_steps >= 0 and ls.max_steps >= 1


def test_the_enumeration_refuses_unsupported_sizes():
    with pytest.raises(ValueError):
        L.all_perms(1)
    with pytest.raises(ValueError):
        L.all_perms(11)
