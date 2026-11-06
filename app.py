"""QAP an einer Mini-Halle - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 1 der Reihe „Anordnen: relaxieren und runden“: die Wurzel. Bauteile mit Flüssen (Netzliste) sollen auf die Plätze eines Rasters; gesucht ist die Anordnung mit den kleinsten Kosten (Fluss mal Abstand). Bei bis zu neun Bauteilen
lassen sich alle Anordnungen aufzählen: das Optimum ist bewiesen, und die Landschaft der Tauschnachbarschaft (lokale Optima, Erfolg eines Abstiegs) liegt offen. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import qmh_constants as C
import qmh_landscape as L
from qmh_evaluation import (
    Settings,
    board_view,
    de,
    descent_view,
    growth_rows,
    hand_view,
    landscape_of,
    landscape_view,
    pair_of,
    pct,
    swap_view,
    verdict_board,
    verdict_core,
    verdict_descent,
    verdict_growth,
    verdict_landscape,
    verdict_swap,
)
from qmh_layout import dunkles_design, kopf_css
from qmh_pdf_export import build_pdf
from qmh_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, randomize_start, sidebar_presets, sync_query_params
from qmh_visualization import build_basins, build_board, build_deltas, build_descent, build_growth, build_hist

st.set_page_config(page_title="QAP an einer Mini-Halle – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Rechne alle Anordnungen …")
def _landscape(seed, n):
    return landscape_of(Settings(seed=seed, n=n))


@st.cache_data(show_spinner="Rechne die Instanzen …")
def _growth(with_nine):
    return growth_rows(extra={9: C.SWEEP_SEEDS_9} if with_nine else None)


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in qmh_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 QAP an einer Mini-Halle")
st.markdown(
    """
**Wohin mit den Bauteilen, wenn die Kosten an den Paaren hängen?** Jedes Netz verbindet Bauteile; je enger zwei verbundene Bauteile sitzen, desto billiger. Das ist das **quadratische Zuordnungsproblem (QAP)**: `n` Bauteile auf `n` Plätze,
Kosten gleich der Summe aus **Fluss mal Abstand** über alle Paare. Bei bis zu neun Bauteilen lassen sich **alle Anordnungen aufzählen**: das Optimum ist dann bewiesen, und man sieht, was eine **Tauschsuche** (zwei Bauteile die Plätze tauschen,
solange es besser wird) kann: wie viele **lokale Optima** es gibt, wo sie liegen und wie oft ein einzelner Abstieg das beste Ergebnis findet. Stück 1 der Reihe „Anordnen: relaxieren und runden“; die Heuristiken selbst (Tabu, Annealing)
stehen in der Fall-Demo [Tor-Zuordnung](https://sebastianhanisch-dockdoor-demo.streamlit.app/) und in der Linie Trajektorien-Metaheuristiken.

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Die Halle:** `n` Plätze auf einem Raster (Abstand = Anzahl der Schritte, waagerecht plus senkrecht), `n` Bauteile (IC, Kondensator, Widerstand), eine Netzliste. Jedes Paar in einem Netz bekommt den Fluss 1.
2. **Die Kosten:** Summe über alle Paare von Fluss mal Abstand der beiden Plätze. Eine **Anordnung** ist eine Zuordnung der Bauteile zu den Plätzen: `n!` Möglichkeiten.
3. **Der Tausch:** zwei Bauteile vertauschen ihre Plätze. Die Änderung der Kosten (`Δ`) steht in einer Formel über die übrigen `n − 2` Bauteile, ohne alles neu zu rechnen.
4. **Das lokale Optimum:** eine Anordnung, bei der kein Tausch die Kosten senkt. Ein **steilster Abstieg** nimmt jeweils den besten Tausch, bis keiner mehr hilft.
5. **Die Bahn:** das Raster hat Spiegelungen und Drehungen; eine Anordnung und ihre Spiegelbilder kosten dasselbe. Die Demo zählt deshalb auch **je Bahn**.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Halle (Experiment 1 bis 4)**")
    n = st.select_slider("Bauteile n", options=C.SIZES, key="n_select", help="Bauteile = Plätze. Bei n = 9 liegen 362 880 Anordnungen vor; die Rechnung läuft auf Knopfdruck (rund eine Viertelminute).")
    seed = st.number_input("Zufalls-Seed der Netzliste", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Netzliste", width="stretch", on_click=randomize_seed)
    st.markdown("**Tausch und Abstieg (Experiment 3 und 4)**")
    start = st.number_input("Zufallsstart Nr.", *bounds("start_input"), key="start_input", step=1, help="Die Startanordnung für Tausch und Abstieg (aus einem festen Zufallsstrom, damit der Link dasselbe zeigt).")
    st.button("🎲 Neuer Start", width="stretch", on_click=randomize_start)
    a = st.select_slider("Bauteil a", options=C.COMPONENT_CHOICES, key="a_select", help="Das erste Bauteil des Tauschs (begrenzt auf n).")
    b = st.select_slider("Bauteil b", options=C.COMPONENT_CHOICES, key="b_select", help="Das zweite Bauteil des Tauschs; sind a und b gleich, rückt b um eins weiter.")

sync_query_params({"seed_input": int(seed), "n_select": int(n), "a_select": int(a), "b_select": int(b), "start_input": int(start)})
settings = Settings(int(seed), int(n), int(a), int(b), int(start))

gated = settings.n == 9 and not st.session_state.get("n9_ok")
if gated:
    st.info("Bei n = 9 sind es 362 880 Anordnungen; die vollständige Rechnung dauert rund eine Viertelminute (auf Streamlit Cloud eher länger). Mit dem Knopf starten:")
    if st.button("⏱️ n = 9 rechnen", key="n9_button"):
        st.session_state["n9_ok"] = True
        st.rerun()
else:
    ls = _landscape(settings.seed, settings.n)
    bv = board_view(settings, ls)
    lv = landscape_view(settings, ls)
    sv = swap_view(settings, ls)
    dv = descent_view(settings, ls)
    names = bv["inst"].names

    # --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

    st.markdown("## 🎯 Ergebnis")
    m1, m2, m3 = st.columns(3)
    m4, m5 = st.columns(2)
    m1.metric("Optimale Kosten", str(ls.opt), help="Die kleinsten Kosten aller n! Anordnungen: aufgezählt, also bewiesen.")
    m2.metric("Mittlere Kosten", de(ls.mean_cost, 1), delta=f"{de(100 * (ls.mean_cost - ls.opt) / ls.opt, 0)} % über dem Optimum", delta_color="off", help="Mittel über alle Anordnungen: so viel kostet eine zufällige Anordnung.")
    m3.metric("Lokale Optima je Bahn", de(ls.local_orbits, 1), help="Schwache lokale Optima der Tauschnachbarschaft, geteilt durch die Zahl der Rasterspiegelungen (Gleichstände erlaubt).")
    m4.metric("Erfolg eines Abstiegs", pct(ls.success, 1), help="Anteil aller Startanordnungen, deren steilster Abstieg ein globales Optimum erreicht (exakt gezählt, nicht gewürfelt).")
    m5.metric("Optimale Anordnungen", f"{ls.n_opt}", help="Wie viele der Anordnungen die optimalen Kosten haben; durch die Rasterspiegelungen mehrfach gezählt.")
    st.plotly_chart(build_board(bv["inst"], bv["best"], f"Eine beste Anordnung (Kosten {ls.opt})"), width="stretch", key="board_chart")
    st.caption("Grau die Plätze, grün die Bauteile auf ihrem Platz, blau die Verbindungen (je dicker, desto mehr Fluss). Gezeigt ist die lexikographisch erste der besten Anordnungen.")

    if st.button("📄 PDF des aktuellen Ergebnisses erstellen", key="pdf_make"):
        st.session_state["pdf_made"] = (settings, build_pdf(settings, bv, ls))
    made = st.session_state.get("pdf_made")
    if made and made[0] == settings:
        st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="qap-mini-halle.pdf", mime="application/pdf", key="pdf_download")

    # --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

    st.markdown("## ❓ Kernfrage: Was kann eine Tauschsuche, wenn man alle Anordnungen kennt?")
    verdict(verdict_core(settings, bv, ls, dv))

    st.markdown("---")

    # --- Experimente ---------------------------------------------------------------------------------------------------------------------------

    st.subheader("🔬 Experiment 1: Eine Mini-Halle")
    st.caption("Die Halle der Seitenleiste: die Netzliste, der daraus entstehende Fluss und die Beiträge der besten Anordnung. Darunter das Handbeispiel mit vier Bauteilen, in dem man alle 24 Anordnungen von Hand nachrechnen kann.")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(pd.DataFrame([{"Netz": i + 1, "Bauteile": ", ".join(names[x] for x in net)} for i, net in enumerate(bv["inst"].nets)]), hide_index=True, width="stretch")
    with c2:
        st.dataframe(pd.DataFrame([{"Paar": f"{names[a_]} – {names[b_]}", "Fluss": f, "Abstand": d, "Beitrag": c} for a_, b_, f, d, c in bv["pairs"]]), hide_index=True, width="stretch")
    verdict(verdict_board(settings, bv, ls))
    hv = hand_view()
    with st.expander("Das Handbeispiel: vier Bauteile, 24 Anordnungen von Hand"):
        st.markdown("Raster 2×2 (Plätze 1 2 / 3 4, Nachbarn haben Abstand 1, die Diagonale 2); Fluss A–B: 3, A–C: 1, B–D: 2, C–D: 4. Eine Anordnung nennt den Platz von A, B, C, D. Die Kosten nehmen genau drei Werte an:")
        st.dataframe(pd.DataFrame([{"Kosten": c, "Zahl der Anordnungen": k, "Beispiel (Plätze von A, B, C, D)": ", ".join(str(x + 1) for x in next(p for cc, p in hv["rows"] if cc == c))} for c, k in hv["values"].items()]),
                     hide_index=True, width="stretch")

    st.markdown("---")

    st.subheader("🔬 Experiment 2: Alle Anordnungen")
    st.caption(f"Kosten aller {C.n_text(ls.perms)} Anordnungen als Histogramm (grün: das Optimum). Spiegelbilder unter den {ls.aut} Rasterautomorphismen kosten dasselbe und kommen deshalb gleich oft vor.")
    st.plotly_chart(build_hist(ls), width="stretch", key="hist_chart")
    verdict(verdict_landscape(settings, lv))

    st.markdown("---")

    st.subheader("🔬 Experiment 3: Der Tausch")
    st.caption("Zwei Bauteile tauschen die Plätze. Die Änderung der Kosten steht in einer Formel über die anderen `n − 2` Bauteile und wird gegen die vollständige Neuberechnung geprüft. Rechts alle möglichen Tausche der Startanordnung.")
    t1, t2 = st.columns(2)
    with t1:
        st.plotly_chart(build_board(sv["inst"], sv["p"], f"Startanordnung Nr. {settings.start} (Kosten {sv['cost_p']})"), width="stretch", key="swap_before")
    with t2:
        st.plotly_chart(build_board(sv["inst"], sv["q"], f"Nach dem Tausch von {names[sv['a']]} und {names[sv['b']]} (Kosten {sv['cost_q']})"), width="stretch", key="swap_after")
    st.plotly_chart(build_deltas(sv["deltas"], names), width="stretch", key="delta_chart")
    verdict(verdict_swap(settings, sv))
    if sv["example"] is not None:
        ex = sv["example"]
        st.plotly_chart(build_board(sv["inst"], ex["p"], f"Ein lokales Optimum, das kein globales ist (Kosten {ex['cost']}, {C.n_text(ex['basin'])} Starts enden hier)"), width="stretch", key="local_example")
        st.caption("Das lokale Optimum mit dem größten Einzugsgebiet unter denen, die kein globales Optimum sind: kein Tausch zweier Bauteile verbessert es.")

    st.markdown("---")

    st.subheader("🔬 Experiment 4: Ein Abstieg und viele Starts")
    st.caption("Der steilste Abstieg von der gewählten Startanordnung (links), und wohin die Abstiege von allen Startanordnungen führen: jedes lokale Optimum mit der Zahl der Starts, die dort enden (rechts).")
    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(build_descent(dv["costs"], ls.opt), width="stretch", key="descent_chart")
    with d2:
        st.plotly_chart(build_basins(ls), width="stretch", key="basin_chart")
    verdict(verdict_descent(settings, dv))

    st.markdown("---")

# --- Experiment 5: Wachstum ----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Wachstum mit der Zahl der Bauteile")
st.caption("Je Größe die Landschaften fester Instanzen (n = 4, 6, 8: 20 Instanzen; n = 9: fünf Instanzen, jede rund eine Viertelminute): Zahl der lokalen Optima je Bahn und Erfolg eines Abstiegs, Median und Spanne.")
if not st.session_state.get("growth_ok"):
    st.info("Die Rechnung für n = 4, 6 und 8 dauert rund eine halbe Minute. Mit dem Knopf starten:")
    if st.button("⏱️ Wachstum rechnen (n = 4, 6, 8)", key="growth_button"):
        st.session_state["growth_ok"] = True
        st.rerun()
else:
    nine = st.session_state.get("growth9_ok", False)
    rows = _growth(nine)
    st.plotly_chart(build_growth(rows), width="stretch", key="growth_chart")
    st.dataframe(pd.DataFrame([{"n": r["n"], "Anordnungen": C.n_text(r["perms"]), "Instanzen": r["seeds"], "lokale Optima je Bahn (Median)": de(r["local_orbits"], 1), "Spanne": f"{de(r['local_min'], 1)} bis {de(r['local_max'], 1)}",
                                "Erfolg eines Abstiegs (Median)": pct(r["success"], 1), "Spanne Erfolg": f"{pct(r['success_min'], 1)} bis {pct(r['success_max'], 1)}", "Tausche je Abstieg (Mittel)": de(r["steps"], 1)} for r in rows]), hide_index=True, width="stretch")
    verdict(verdict_growth(rows))
    if not nine:
        if st.button("⏱️ n = 9 dazu (fünf Instanzen, rund eine Minute)", key="growth9_button"):
            st.session_state["growth9_ok"] = True
            st.rerun()

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Es gibt nur wenige Bauteile** | Die Aufzählung wächst mit `n!`: bei zwölf Bauteilen sind es 479 001 600 Anordnungen. Die Landschaft ist dann nicht mehr vollständig zu sehen; Aussagen über lokale Optima und den Erfolg eines Abstiegs gelten nur bis `n = 9`. | spätere Stücke der Reihe (Schranken, Relaxation) |
| **Die Familie „Netzliste“** | Alle Zahlen gelten für Netzlisten aus Dreiergruppen und Raster mit Manhattan-Abstand. Bei einzelnen anderen Flussfamilien (etwa zwei Cluster) liegt der Erfolg eines Abstiegs deutlich anders, bei anderen kaum (Stück 2); die Demo misst nur diese Familie. | Stück 2 der Reihe |
| **Der Tausch zweier Bauteile ist die einzige Nachbarschaft** | Andere Nachbarschaften (drei Bauteile im Kreis, Verschieben ganzer Gruppen) haben andere lokale Optima; die Zahlen hier gelten nur für den Paartausch. | Trajektorien-Metaheuristiken |
| **Es zählt, was die Aufzählung zeigt** | Metaheuristiken wie Tabu-Suche oder Annealing laufen weiter, wenn die Tauschsuche stehenbleibt. Wie gut sie das lösen, zeigt diese Demo nicht. | Fall-Demo Tor-Zuordnung, Trajektorien-Metaheuristiken |
| **Gleiche Plätze, symmetrisches Raster** | Die Bahnen teilen durch die Zahl der Rasterspiegelungen (2×2 und 3×3: 8, die anderen Raster: 4). Haben zwei Bauteile genau denselben Fluss zu allen anderen, gibt es zusätzliche Gleichstände, die die Zählung je Bahn nicht erfasst. | Stück 13 (Symmetrie) |
| **Ganzzahlige Flüsse und Abstände** | Gleichstände sind häufig; deshalb zählt die Demo schwache lokale Optima (ohne strikten Gewinn) als Hauptzahl und nennt strenge Optima getrennt. Mit reellen Flüssen wären Gleichstände zwischen Nachbarn vermutlich selten (nicht gemessen). | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Das Problem (Koopmans–Beckmann-Form).** Gegeben sind ein symmetrischer Fluss $F \in \mathbb{N}^{n \times n}$ zwischen den Bauteilen und ein symmetrischer Abstand $D \in \mathbb{N}^{n \times n}$ zwischen den Plätzen, beide mit Diagonale 0.
Gesucht ist eine Permutation $p$ (Bauteil $i$ sitzt auf Platz $p_i$), die
$$\mathrm{cost}(p) = \sum_{i<k} F_{ik}\, D_{p_i p_k}$$
minimiert. Es gibt $n!$ Anordnungen.

**Die Lawler-Form.** Mit der Kostenmatrix $C_{(i,j),(k,l)} = F_{ik}\, D_{jl}$ ist $\mathrm{cost}(p) = \sum_{i<k} C_{(i,p_i),(k,p_k)}$: dieselbe Aufgabe mit $n^2 \times n^2$ Zahlen. Die Demo prüft beide Formen gegeneinander.

**Der Tausch.** Vertauscht man die Bauteile $a$ und $b$, ändert sich nur, was an $a$ oder $b$ hängt:
$$\Delta(p; a, b) = \sum_{k \ne a, b} (F_{ak} - F_{bk})\,\big(D_{p_b p_k} - D_{p_a p_k}\big).$$
Das Paar $(a, b)$ selbst ändert sich nicht, weil $D$ symmetrisch ist. Ein Tausch kostet $O(n)$ statt $O(n^2)$.

**Lokale Optima.** Eine Anordnung ist ein (schwaches) lokales Optimum der Tauschnachbarschaft, wenn $\Delta(p; a, b) \ge 0$ für alle $n(n-1)/2$ Paare gilt, ein strenges, wenn $\Delta > 0$ für alle. Der **steilste Abstieg** tauscht das Paar mit kleinstem $\Delta$ (bei Gleichstand das erste in der Reihenfolge der Paare), solange $\Delta < 0$.

**Bahnen.** Eine Permutation $\sigma$ der Plätze, die alle Abstände erhält ($D_{\sigma(j)\sigma(l)} = D_{jl}$), bildet jede Anordnung auf eine gleich gute ab. Die Gruppe dieser Spiegelungen und Drehungen hat 4 oder 8 Elemente; die Zahlen „je Bahn“ sind die Zahlen der Anordnungen geteilt durch diese Zahl.

**Literatur.** T. C. Koopmans, M. Beckmann: *Assignment problems and the location of economic activities*, Econometrica 25 (1957), und E. L. Lawler: *The quadratic assignment problem*, Management Science 9 (1963), für die beiden Formen (Angaben aus der Fachliteratur übernommen, hier nicht nachgelesen);
R. E. Burkard, E. Çela, P. M. Pardalos, L. S. Pitsoulis: *The quadratic assignment problem*, im Handbook of Combinatorial Optimization (1998), als Überblick (nicht eingesehen). Die Zahlen dieser Demo sind eigene Rechnungen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
