"""Orakel (ORACLE-PLAYBOOK §5): die vollständige Aufzählung ist das Orakel für alles; `scipy.optimize.quadratic_assignment` (Methode „2opt“) darf das Optimum nie unterschreiten und trifft es mit genügend Starts.
Ohne scipy (nur requirements-dev.txt) werden die Tests übersprungen."""

import pytest

import qmh_instance as I
import qmh_landscape as L

scipy_opt = pytest.importorskip("scipy.optimize")


def _scipy_cost(F, D, seed):
    """Koopmans–Beckmann-Kosten, die scipy minimiert (Fluss A, Abstand B; ganze Summe, also das Doppelte der Paarkosten)."""
    res = scipy_opt.quadratic_assignment(F.astype(float), D.astype(float), method="2opt", options={"rng": seed, "partial_guess": None, "maximize": False})
    return res.fun / 2.0


@pytest.mark.parametrize("n", (4, 6, 8))
def test_the_two_opt_solver_never_beats_the_enumeration_and_finds_it_with_restarts(n):
    for seed in range(4):
        inst = I.make_instance(n, seed)
        opt = L.landscape(inst).opt
        found = [_scipy_cost(inst.F, inst.D, s) for s in range(40)]
        assert min(found) >= opt - 1e-9 and min(found) == pytest.approx(opt)
