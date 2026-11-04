"""Pfade der App, die der Rauchtest nicht erreicht: die fünf Experimente im Seiteninhalt, PDF, jede Größe, die Rechenknöpfe, Eckfälle (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _widget(at, key):
    if key in ("seed_input", "start_input"):
        return at.sidebar.number_input(key=key)
    return at.sidebar.select_slider(key=key)


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        _widget(at, key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_the_metrics_and_the_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert values["Optimale Kosten"] == "31" and values["Mittlere Kosten"] == "50,0" and values["Lokale Optima je Bahn"] == "28,0" and values["Erfolg eines Abstiegs"] == "26,6 %" and values["Optimale Anordnungen"] == "4"
    text = _texts(at)
    for part in ("Die beste von 40 320 Anordnungen kostet **31**", "112** lokale Optima der Tauschnachbarschaft", "stimmt mit der Neuberechnung überein", "endet nach 3 Tauschen bei Kosten 31", "Mit dem Knopf starten"):
        assert part in text, part


def test_the_five_experiments_the_limits_table_and_the_formulation_are_on_the_page():
    at = _app()
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬")] == [f"🔬 Experiment {i}" for i in range(1, 6)] and "🚧 Wo die Annahmen enden" in headings
    assert [e.label for e in at.expander] == ["So funktioniert es", "Das Handbeispiel: vier Bauteile, 24 Anordnungen von Hand", "📐 Mathematische Formulierung"]
    assert not at.error and len(at.get("plotly_chart")) == 8


def test_small_sizes_render_their_landscape_and_nine_waits_for_the_button():
    for n, perms in ((4, "24"), (6, "720")):
        at = _app(n_select=n)
        assert f"Die beste von {perms} Anordnungen" in _texts(at)
    at9 = _app(n_select=9)
    assert any("362 880 Anordnungen" in str(i.value) for i in at9.info) and any(b.label == "⏱️ n = 9 rechnen" for b in at9.button)
    assert not at9.metric and "Die beste von" not in _texts(at9)
    at9.button(key="n9_button").click().run()
    assert not at9.exception and "Die beste von 362 880 Anordnungen" in _texts(at9)


def test_the_pdf_button_makes_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert any(d.proto.label == "⬇️ PDF herunterladen" for d in at.get("download_button"))


def test_the_growth_button_computes_the_table_for_four_six_and_eight():
    at = _app()
    assert any(b.label == "⏱️ Wachstum rechnen (n = 4, 6, 8)" for b in at.button)
    at.button(key="growth_button").click().run()
    assert not at.exception
    assert any("n = 8: 28 lokale Optima je Bahn" in str(e.value) for e in at.success) and any(b.label.startswith("⏱️ n = 9 dazu") for b in at.button)


def test_the_swap_and_start_controls_change_the_verdicts():
    at = _app(a_select=3, b_select=6)
    assert "Der Tausch von R1 und R2" in _texts(at)
    at2 = _app(start_input=7)
    assert "Start Nr. 7" in _texts(at2)


def test_equal_components_a_and_b_are_moved_apart():
    at = _app(a_select=4, b_select=4)
    assert not at.exception and "Der Tausch von IC2 und C2" in _texts(at)


def test_a_larger_component_choice_than_n_is_clamped():
    at = _app(n_select=4, a_select=9, b_select=8)
    assert not at.exception and "Der Tausch von" in _texts(at)
