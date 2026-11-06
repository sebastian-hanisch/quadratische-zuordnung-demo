"""Verdikte (reine Funktionen): Art und Text an handgesetzten Werten und an den kleinen Landschaften; jede Zahl im Text stammt aus den Werten."""

import qmh_constants as C
import qmh_evaluation as ev
import qmh_landscape as L


def _rows(local, success, n=(4, 6, 8)):
    return [{"n": k, "perms": __import__("math").factorial(k), "seeds": 20, "local_orbits": lo, "local_min": lo - 1, "local_max": lo + 2, "success": su, "success_min": su - 0.1, "success_max": su + 0.1, "steps": 2.0, "opt_orbits": 1.0}
            for k, lo, su in zip(n, local, success)]


def test_growth_verdict_claims_growth_only_when_it_holds():
    kind, text = ev.verdict_growth(_rows((1.0, 6.0, 22.0), (0.9, 0.76, 0.37)))
    assert kind == "success" and "Im Median wächst die Zahl der lokalen Optima mit n und der Erfolg eines einzelnen Abstiegs sinkt" in text and "n = 6: 6 lokale Optima je Bahn, Erfolg eines Starts 76 %" in text and "24 auf 40 320" in text
    kind, text = ev.verdict_growth(_rows((1.0, 6.0, 5.0), (0.9, 0.76, 0.8)))
    assert kind == "info" and "nicht gleichmäßig" in text


def test_board_verdict_names_the_orbits_and_the_count(hand):
    ls = L.landscape(hand)
    v = ev.board_view(ev.Settings(n=4), L.landscape(ev.make_instance(4, 35)))
    kind, text = ev.verdict_board(ev.Settings(n=4), v, L.landscape(ev.make_instance(4, 35)))
    assert kind == "success" and "eine Bahn" in text and "Optimum ist bewiesen" in text and "24 Anordnungen" in text
    assert ls.n_opt == 8


def test_landscape_verdict_is_a_warning_exactly_when_a_local_optimum_is_not_global(land6, hand):
    inst, ls = land6
    s = ev.Settings(n=6)
    kind, text = ev.verdict_landscape(s, ev.landscape_view(s, ls))
    assert kind == "warning" and f"**{ls.n_local}**" in text and f"{ls.n_local_strict} streng" in text and "kein globales Optimum" in text
    kind, text = ev.verdict_landscape(ev.Settings(n=4), ev.landscape_view(ev.Settings(n=4), L.landscape(hand)))
    assert kind == "success" and "0 lokale Optima sind kein globales Optimum" in text


def test_swap_verdict_confirms_the_formula_and_names_the_components():
    s = ev.Settings(n=6, a=3, b=6, start=1)
    ls = ev.landscape_of(s)
    v = ev.swap_view(s, ls)
    kind, text = ev.verdict_swap(s, v)
    assert kind == "success" and v["delta_formula"] == v["delta_recomputed"] and f"{v['inst'].names[2]} und {v['inst'].names[5]}" in text and "stimmt mit der Neuberechnung überein" in text
    bad = dict(v, delta_formula=v["delta_formula"] + 1)
    kind, text = ev.verdict_swap(s, bad)
    assert kind == "error" and "WEICHT" in text


def test_swap_verdict_for_a_local_optimum_and_for_a_zero_change(land6):
    inst, ls = land6
    s = ev.Settings(n=6)
    v = ev.swap_view(s, ls)
    local = dict(v, is_local=True, improving=0)
    assert "ist ein lokales Optimum" in ev.verdict_swap(s, local)[1]
    flat = dict(v, delta_formula=0, delta_recomputed=0, cost_q=v["cost_p"])
    assert "ändert nicht" in ev.verdict_swap(s, flat)[1] and "Δ = 0" in ev.verdict_swap(s, flat)[1]


def test_descent_verdict_names_global_or_local_and_the_success(land6):
    inst, ls = land6
    s = ev.Settings(n=6, start=1)
    dv = ev.descent_view(s, ls)
    kind, text = ev.verdict_descent(s, dv)
    assert kind == ("success" if dv["global"] else "warning") and f"{ls.success * 100:.1f}".replace(".", ",") in text and "Zufallsstarts" in text
    trap = dict(dv, **{"global": False})
    assert ev.verdict_descent(s, trap)[0] == "warning" and "kein globales ist" in ev.verdict_descent(s, trap)[1]


def test_core_verdict_states_the_counts(land8):
    inst, ls = land8
    s = ev.Settings(n=8)
    v = ev.board_view(s, ls)
    kind, text = ev.verdict_core(s, v, ls, ev.descent_view(s, ls))
    assert kind == "info" and "40 320" in text and "479 001 600" in text and f"{ls.n_local} von" in text


def test_number_formatting():
    assert ev.de(-1.5, 1) == "−1,5" and ev.pct(0.266, 1) == "26,6 %" and ev.pct(0.5) == "50 %" and C.n_text(362880) == "362 880"
