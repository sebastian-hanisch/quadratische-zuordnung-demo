"""Kosten, Tausch-Delta und Lawler-Form: Handwerte, Eigenschaften und Randfälle (DEMO-PLAYBOOK §1b, ORACLE-PLAYBOOK §5)."""

from itertools import permutations

import numpy as np
import pytest

import qmh_cost as K
import qmh_instance as I


def test_the_distances_of_the_2x2_grid(hand):
    assert hand.D.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 1], [2, 1, 1, 0]]


def test_the_hand_costs(hand):
    """README: Identität 3·1 + 1·1 + 2·1 + 4·1 = 10; Tausch von A und D (p = (3, 1, 2, 0)) ebenfalls 10 (Δ = 0); p = (1, 2, 0, 3): 6 + 1 + 2 + 8 = 17."""
    assert K.cost([0, 1, 2, 3], hand.F, hand.D) == 10
    assert K.cost([3, 1, 2, 0], hand.F, hand.D) == 10 and K.swap_delta([0, 1, 2, 3], 0, 3, hand.F, hand.D) == 0
    assert K.cost([1, 2, 0, 3], hand.F, hand.D) == 17


def test_the_hand_example_has_three_cost_values_eight_times_each(hand):
    costs = sorted(K.cost(p, hand.F, hand.D) for p in permutations(range(4)))
    assert costs == [10] * 8 + [13] * 8 + [17] * 8


def test_pair_costs_sum_to_the_cost_and_are_sorted(hand):
    rows = K.pair_costs([1, 2, 0, 3], hand.F, hand.D)
    assert sum(r[4] for r in rows) == 17 and [r[4] for r in rows] == sorted((r[4] for r in rows), reverse=True) and rows[0][:4] == (2, 3, 4, 2)


def test_the_swap_delta_equals_the_recomputed_difference_for_all_sizes():
    rng = np.random.default_rng(3)
    for n in (4, 6, 8, 9):
        for seed in range(6):
            inst = I.make_instance(n, seed)
            p = rng.permutation(n).tolist()
            for a, b in K.pair_list(n):
                q = K.swapped(p, a, b)
                assert K.swap_delta(p, a, b, inst.F, inst.D) == K.cost(q, inst.F, inst.D) - K.cost(p, inst.F, inst.D)


def test_the_vectorised_deltas_equal_the_single_formula(land6):
    inst, _ = land6
    from qmh_landscape import all_perms
    P = all_perms(6)[::37].astype(np.intp)
    dl = K.all_deltas(P, inst.F, inst.D)
    for r in range(0, len(P), 5):
        for j, (a, b) in enumerate(K.pair_list(6)):
            assert dl[r, j] == K.swap_delta(P[r].tolist(), a, b, inst.F, inst.D)


def test_lawler_form_equals_koopmans_beckmann_form():
    for n in (4, 6):
        inst = I.make_instance(n, 11)
        for p in list(permutations(range(n)))[::13]:
            assert K.lawler_cost(p, inst.F, inst.D) == K.cost(p, inst.F, inst.D)


def test_the_lawler_matrix_by_hand_for_three_components():
    F = np.array([[0, 2, 1], [2, 0, 0], [1, 0, 0]], dtype=np.int16)
    D = I.grid_distance(1, 3)
    C = K.lawler_matrix(F, D)
    assert C.shape == (3, 3, 3, 3) and C[0, 0, 1, 2] == 2 * 2 and C[0, 1, 2, 2] == 1 * 1 and K.cost([1, 0, 2], F, D) == 2 * 1 + 1 * 1


def test_the_cost_is_invariant_under_the_grid_automorphisms():
    import qmh_landscape as L
    inst = I.make_instance(6, 5)
    autos = L.automorphisms(inst.rows, inst.cols)
    p = [3, 0, 5, 1, 4, 2]
    base = K.cost(p, inst.F, inst.D)
    for sigma in autos:
        assert K.cost([sigma[x] for x in p], inst.F, inst.D) == base


def test_cost_rejects_a_non_permutation(hand):
    with pytest.raises(ValueError):
        K.cost([0, 0, 1, 2], hand.F, hand.D)
    with pytest.raises(ValueError):
        K.swap_delta([0, 1, 2, 3], 1, 1, hand.F, hand.D)
