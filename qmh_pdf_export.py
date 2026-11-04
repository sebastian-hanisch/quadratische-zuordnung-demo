"""PDF-Export des aktuellen Ergebnisses (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from qmh_evaluation import Settings, de

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "Δ": "Delta", "↔": "<->", "„": '"', "“": '"', "²": "2", "³": "3", "⁻": "-", "¹": "1"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: Settings, v: dict, ls) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("QAP an einer Mini-Halle"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{s.n} Bauteile auf einem {v['inst'].rows}x{v['inst'].cols}-Raster, Seed {s.seed}; alle {ls.perms} Anordnungen aufgezählt. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    names = v["inst"].names
    rows = [("Optimale Kosten", ls.opt), ("Mittlere Kosten aller Anordnungen", de(ls.mean_cost, 1)), ("Schlechteste Kosten", int(ls.cost_values[-1])), ("Optimale Anordnungen (Bahnen)", f"{ls.n_opt} ({de(ls.optima_orbits, 1)})"),
            ("Lokale Optima der Tauschnachbarschaft (je Bahn)", f"{ls.n_local} ({de(ls.local_orbits, 1)})"), ("davon streng", ls.n_local_strict),
            ("Erfolg eines Abstiegs (Anteil der Starts)", de(100 * ls.success, 1) + " %"), ("Platz des Bauteils (beste Anordnung)", ", ".join(f"{names[i]}: {v['best'][i] + 1}" for i in range(len(v["best"]))))]
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(100, 6, "Kennzahl", border=1)
    pdf.cell(85, 6, "Wert", border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for k, val in rows:
        pdf.cell(100, 5.5, _s(k), border=1)
        pdf.cell(85, 5.5, _s(val), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: qap-mini-halle-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
