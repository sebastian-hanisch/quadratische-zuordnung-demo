"""Instanz: Netzliste, Fluss, Raster, SplitMix64 und die Startanordnungen (deterministisch auf jedem Rechner)."""

import numpy as np
import pytest

import qmh_constants as C
import qmh_instance as I
from qmh_rng import SplitMix64


def test_splitmix64_reference_values():
    """Die ersten Ausgaben von SplitMix64 mit Saat 0 (Referenzwerte der Veröffentlichung von Vigna)."""
    r = SplitMix64(0)
    assert [r.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_instances_are_deterministic_and_seed_dependent():
    a, b, c = I.make_instance(8, 35), I.make_instance(8, 35), I.make_instance(8, 36)
    assert a.nets == b.nets and np.array_equal(a.F, b.F) and a.nets != c.nets


@pytest.mark.parametrize("n", C.SIZES)
def test_the_flow_is_symmetric_integer_with_zero_diagonal_and_the_grid_fits(n):
    for seed in (0, 1, 35, 999999):
        inst = I.make_instance(n, seed)
        assert (inst.F == inst.F.T).all() and (np.diag(inst.F) == 0).all() and (inst.F >= 0).all()
        assert inst.rows * inst.cols == n and inst.D.shape == (n, n) and (inst.D == inst.D.T).all() and (np.diag(inst.D) == 0).all()
        assert len(inst.names) == n and len(set(inst.names)) == n and len(inst.nets) == max(2, int(round(1.5 * n)))
        assert all(2 <= len(net) <= min(4, n) and len(set(net)) == len(net) and all(0 <= x < n for x in net) for net in inst.nets)


def test_the_flow_counts_the_shared_nets_by_hand():
    F = I.flow_from_nets(4, ((0, 1, 2), (1, 2), (0, 3)))
    assert F.tolist() == [[0, 1, 1, 1], [1, 0, 2, 0], [1, 2, 0, 0], [1, 0, 0, 0]]


def test_component_names_follow_the_groups_of_three():
    assert I.component_names(8) == ("IC1", "C1", "R1", "IC2", "C2", "R2", "IC3", "C3")


def test_grid_distance_by_hand_and_rejections():
    assert I.grid_distance(2, 3).tolist() == [[0, 1, 2, 1, 2, 3], [1, 0, 1, 2, 1, 2], [2, 1, 0, 3, 2, 1], [1, 2, 3, 0, 1, 2], [2, 1, 2, 1, 0, 1], [3, 2, 1, 2, 1, 0]]
    with pytest.raises(ValueError):
        I.grid_distance(0, 3)
    with pytest.raises(ValueError):
        I.make_instance(7, 1)


def test_random_starts_are_permutations_deterministic_and_vary():
    a, b = I.random_start(8, 1), I.random_start(8, 1)
    assert a == b and sorted(a) == list(range(8))
    assert len({tuple(I.random_start(8, k)) for k in range(1, 30)}) > 20


def test_the_hand_instance_matches_the_readme(hand):
    assert hand.n == 4 and hand.F[0, 1] == 3 and hand.F[2, 3] == 4 and hand.F.sum() == 2 * (3 + 1 + 2 + 4)
