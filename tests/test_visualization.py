"""Abbildungen: sie entstehen für jede Eingabe, Achsen sind gesperrt, Beschriftungen und Daten stimmen mit den Werten überein; PDF-Export."""

import qmh_constants as C
import qmh_evaluation as ev
import qmh_visualization as V
from qmh_pdf_export import build_pdf


def _locked(fig):
    return fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange


def test_board_chart_has_a_marker_and_a_label_for_every_component(land6):
    inst, ls = land6
    fig = V.build_board(inst, [int(x) for x in ls.optima[0]], "Titel")
    comp = fig.data[-1]
    assert list(comp.text) == list(inst.names) and len(comp.x) == 6 and len(fig.data[0].x) == 6 and fig.layout.title.text == "Titel" and _locked(fig)
    assert {t.name for t in fig.data if t.name} == {f"Fluss {w}" for w in sorted({int(inst.F[a, b]) for a in range(6) for b in range(a + 1, 6) if inst.F[a, b]})}
    assert tuple(fig.layout.xaxis.range) == (-0.7, inst.cols - 0.3)                    # die Achsen haben feste Bereiche (ein Anker an die andere Achse hatte sie zerstört)


def test_board_chart_for_every_size_and_the_hand_example(hand):
    for n in C.SIZES:
        inst = ev.make_instance(n, 3)
        fig = V.build_board(inst, list(range(n)), f"n = {n}")
        assert len(fig.data[-1].x) == n
    assert len(V.build_board(hand, [0, 1, 2, 3], "Hand").data[-1].x) == 4


def test_histogram_marks_the_optimum_and_counts_all_arrangements(land6):
    _, ls = land6
    fig = V.build_hist(ls)
    bar = fig.data[0]
    assert int(sum(bar.y)) == 720 and bar.marker.color[0] == C.COLORS["best"] and all(c == C.COLORS["other"] for c in bar.marker.color[1:]) and _locked(fig)


def test_delta_chart_colours_the_improving_swaps(land6):
    inst, ls = land6
    s = ev.Settings(n=6)
    sv = ev.swap_view(s, ls)
    fig = V.build_deltas(sv["deltas"], inst.names)
    ys = list(fig.data[0].y)
    assert ys == sorted(ys) and len(ys) == 15 and sum(1 for y in ys if y < 0) == sv["improving"] == sum(1 for c in fig.data[0].marker.color if c == C.COLORS["best"])


def test_descent_chart_and_basin_chart(land6):
    _, ls = land6
    s = ev.Settings(n=6)
    dv = ev.descent_view(s, ls)
    fig = V.build_descent(dv["costs"], ls.opt)
    assert list(fig.data[0].y) == dv["costs"] and _locked(fig)
    bf = V.build_basins(ls)
    assert sum(len(t.x) for t in bf.data) == ls.n_local and bf.layout.yaxis.type == "log" and _locked(bf)


def test_growth_chart_survives_a_single_instance_and_a_range():
    rows = ev.growth_rows(sizes=(4, 6), seeds=(1, 2))
    fig = V.build_growth(rows)
    assert len(fig.data) == 2 and list(fig.data[0].x) == [4, 6] and _locked(fig)


def test_the_pdf_export_returns_a_pdf_for_two_sizes(land6, land8):
    for inst, ls in (land6, land8):
        s = ev.Settings(n=inst.n)
        data = build_pdf(s, ev.board_view(s, ls), ls)
        assert data[:5] == b"%PDF-" and len(data) > 1500
