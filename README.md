# QAP an einer Mini-Halle – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-quadratische-zuordnung-demo.streamlit.app/)** -->

Stück 1 von 15 der **Konzepte-Reihe „Anordnen: relaxieren und runden“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Reihe ist in Planung, die Konzepte-Seite gibt es noch nicht). Wie jede Konzepte-Demo dreht sich alles um **eine** Aufgabe mit wachsendem Beispiel: Bauteile mit Flüssen auf die Plätze eines Rasters. Dieses Stück ist die **Wurzel**: Es baut keinen Löser und keine Schranke, sondern zeigt die Aufgabe und ihre Tauschlandschaft **vollständig**, solange man alle Anordnungen aufzählen kann (bis neun Bauteile). Spätere Stücke bringen Schranken, Relaxationen und die Platzierung.

## Warum dieses Problem

Je enger zwei Bauteile sitzen, die ein Netz verbindet, desto billiger; die Kosten hängen also an **Paaren**. Das ist das **quadratische Zuordnungsproblem (QAP)**: `n` Bauteile auf `n` Plätze, gesucht ist die Anordnung mit der kleinsten Summe aus Fluss mal Abstand. Eine einfache Suche tauscht zwei Bauteile, solange es besser wird, und bleibt in einem **lokalen Optimum** stehen. Wie viele es davon gibt und wie oft ein einzelner Abstieg das Beste findet, sieht man nur, wenn man **alle** Anordnungen kennt. Die Kernfrage: **Was kann eine Tauschsuche, wenn man die ganze Landschaft sieht?**

Die Heuristiken selbst (Tabu-Suche, Annealing und andere) stehen nicht hier: die Fall-Demo [Tor-Zuordnung](https://github.com/sebastian-hanisch/dock_door-demo) vergleicht sie an einem QAP mit Toren, die Linie Trajektorien-Metaheuristiken zeigt sie an einer Lieferroute. Hier geht es allein um die Landschaft kleiner Instanzen.

## Modell

- **Instanz** (`qmh_instance.py`): `n` Bauteile (IC, Kondensator, Widerstand in Dreiergruppen) und eine Netzliste mit etwa `1,5 n` Netzen aus zwei bis vier Bauteilen (70 % innerhalb einer Dreiergruppe); jedes Paar in einem Netz zählt den Fluss 1 (ganzzahlig, symmetrisch). Die Plätze liegen auf einem Raster (4: 2×2, 6: 2×3, 8: 2×4, 9: 3×3), der Abstand ist der Manhattan-Abstand. Alle Zufallsentscheidungen laufen über SplitMix64 (`qmh_rng.py`, wie in der Leiterplatte-Demo); die Zahlen sind unter Windows und in der Linux-CI gleich.
- **Kosten** (`qmh_cost.py`): Summe über alle Paare von Fluss mal Abstand der Plätze (Koopmans–Beckmann-Form); die Lawler-Form (volle `n² × n²`-Kostenmatrix) dient als Gegenprobe. Der **Tausch** zweier Bauteile ändert die Kosten um `Δ = Σ_{k≠a,b} (F[a,k] − F[b,k]) · (D[p_b,p_k] − D[p_a,p_k])`, in `O(n)` statt `O(n²)`.
- **Landschaft** (`qmh_landscape.py`): alle `n!` Anordnungen, die Kosten jeder, die **schwachen** lokalen Optima der Tauschnachbarschaft (kein Tausch senkt die Kosten; Gleichstände erlaubt), die **strengen** (jeder Tausch erhöht sie) und der **steilste Abstieg** von jeder Anordnung (der Tausch mit dem kleinsten `Δ`, bei Gleichstand der erste in der Reihenfolge der Paare, solange `Δ < 0`); daraus der **Erfolg eines Abstiegs**: der Anteil aller Startanordnungen, die ein globales Optimum erreichen, **exakt gezählt**, nicht gewürfelt.
- **Bahnen:** das Raster hat Spiegelungen und Drehungen (2×2 und 3×3: acht, 2×3, 2×4: vier); eine Anordnung und ihre Spiegelbilder kosten dasselbe. Die Demo nennt Optima und lokale Optima auch **je Bahn** (die Zahl geteilt durch die der Spiegelungen).
- **Experimente:** (1) eine Mini-Halle und das Handbeispiel; (2) alle Anordnungen; (3) der Tausch; (4) ein Abstieg und viele Starts; (5) das Wachstum mit `n`. Formeln im 📐-Expander der App.

## Methodik

Fünf Experimente mit live berechnetem Verdikt. Der steilste Abstieg ist hier **ein Werkzeug, kein Verfahren, das verglichen wird**: er zeigt, wohin eine Tauschsuche führt. Gerechnet wird in numpy: je Instanz einmal alle Anordnungen und für jede alle `n(n−1)/2` Tausche, bei `n = 8` in etwa einer Sekunde, bei `n = 9` in rund einer Viertelminute (Zeiten hängen vom Rechner ab und werden nicht getestet); `n = 9` und das Wachstum laufen auf Knopfdruck.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (die Aufzählung ist deterministisch, die Zahlen sind **exakt**, nicht Bänder; Zeiten nie). Familie: Netzliste aus Dreiergruppen, Raster, Manhattan-Abstand, Fluss 1 je Paar.

| Frage | Befund |
|---|---|
| Handbeispiel `n = 4` (Raster 2×2; Fluss A–B 3, A–C 1, B–D 2, C–D 4) | 24 Anordnungen, Kosten **10, 13 und 17, je achtmal**; Identität: `3·1 + 1·1 + 2·1 + 4·1 = 10`; Tausch von A und D: ebenfalls 10 (`Δ = 0`); Anordnung (Plätze 2, 3, 1, 4): `6 + 1 + 2 + 8 = 17`; alle 8 Optima sind schwache lokale Optima, keines ist streng, eine Bahn |
| Standardfall `n = 8`, Seed 35 | 40 320 Anordnungen; Optimum **31** (4 Anordnungen, eine Bahn), schlechteste 69, Mittel 50,0; **112** schwache lokale Optima (**28,0 je Bahn**), davon 8 streng, **108 sind kein globales Optimum**; Erfolg eines Abstiegs **26,6 %** (10 728 von 40 320 Starts), im Mittel 3,8 Zufallsstarts bis zum ersten Treffer; Abstiege dauern im Mittel 2,97 Tausche, höchstens 6 |
| Tausch und Abstieg im Standardfall | Start Nr. 1 kostet 45; Tausch von IC1 und C1: Kosten 47 (`Δ = +2`, Formel gleich Neuberechnung); 7 der 28 Tausche verbessern, der beste hat `Δ = −10`; der Abstieg 45 → 35 → 32 → 31 endet nach 3 Tauschen im Optimum (Einzugsgebiet dieses Endpunkts: 2 682 Starts); das lokale Optimum mit dem größten Einzugsgebiet unter den nicht globalen kostet 32 (817 Starts) |
| `n = 6`, Seed 35 | 720 Anordnungen, Optimum 26 (8 Anordnungen, 2 Bahnen), Mittel 33,3, schlechteste 41; 12 lokale Optima (3,0 je Bahn), 4 streng, 4 davon kein globales Optimum; Erfolg eines Abstiegs 81,1 % (584 von 720) |
| `n = 4`, Seed 35 | Kosten nur zwei Werte: 26 (8 Anordnungen) und 27 (16): bei vier Bauteilen ist die Landschaft flach |
| Wachstum, Median über feste Instanzen (20 für `n` = 4, 6, 8; fünf für `n = 9`) | **lokale Optima je Bahn: 1,5 / 7,0 / 28,0 / 41,0** (Spannen 1,0 bis 3,0; 2,0 bis 16,0; 12,0 bis 63,0; 36,0 bis 66,0) bei `n` = 4 / 6 / 8 / 9; **Erfolg eines Abstiegs: 100 % / 81,4 % / 35,9 % / 26,6 %** (Spannen 28,9 bis 100 % bei `n = 6`, 11,6 bis 81,6 % bei `n = 8`, 17,6 bis 52,4 % bei `n = 9`); die Zahl der Anordnungen wächst von 24 auf 362 880 |
| `n = 9`, Seed 100000 (Preset) | 362 880 Anordnungen, Optimum 82 (8 Anordnungen, eine Bahn), 288 lokale Optima (36,0 je Bahn), davon 40 streng, 280 kein globales Optimum, Erfolg eines Abstiegs 17,6 % |
| Bahnen | Rasterautomorphismen: 2×2 und 3×3: 8, 2×3 und 2×4: 4; die Zahl der Optima ist ohne Zwillingsbauteile ein Vielfaches davon, und der Tausch von **Zwillingen** (gleicher Fluss zu allen anderen) ändert die Kosten nie |
| Formeln | Tausch-`Δ` gleich der Differenz der Neuberechnung (alle Paare, vier Größen, sechs Instanzen); die vektorisierten `Δ` gleich der Einzelformel; Lawler-Form gleich Koopmans–Beckmann-Form; unabhängiger Löser (`scipy`, „2opt“ mit 40 Starts) unterschreitet das aufgezählte Optimum nie und trifft es bei `n` = 4, 6, 8 |

**Was die Zahlen nicht sagen:** der Standardfall (Seed 35) ist **eine** Instanz, nicht typisch; die Spannen bei `n = 8` (Erfolg 11,6 bis 81,6 %) sind größer als der Unterschied zwischen manchen Größen. Der Median bei `n = 9` beruht auf nur **fünf** Instanzen. Die Seed-35-Instanz mit neun Bauteilen hätte 320 optimale Anordnungen und einen Erfolg von 92,6 %: deshalb nimmt das Preset einen anderen, typischeren Seed.

## Ehrliche Grenzen

- **Wenige Bauteile:** die Aufzählung wächst mit `n!`; bei zwölf Bauteilen sind es 479 001 600 Anordnungen. Alle Aussagen über lokale Optima und den Erfolg eines Abstiegs gelten bis `n = 9`.
- **Eine Familie:** Netzlisten aus Dreiergruppen mit Raster-Abständen. In Vorabmessungen (anderer Zufallsstrom) lag der Erfolg eines Abstiegs bei zufälligen, gruppierten oder sternförmigen Flüssen anders; die Demo misst nur diese Familie (Stück 2 der Reihe).
- **Eine Nachbarschaft:** der Paartausch. Andere Nachbarschaften haben andere lokale Optima.
- **Keine Heuristik gemessen:** Tabu-Suche und Annealing laufen weiter, wenn die Tauschsuche stehenbleibt; wie gut sie das lösen, zeigt diese Demo nicht.
- **Ganzzahlige Flüsse:** Gleichstände sind häufig, deshalb zählt die Demo schwache lokale Optima als Hauptzahl; mit reellen Flüssen wären sie selten.
- **Zwillinge und Symmetrie:** die Zählung je Bahn berücksichtigt nur die Rasterspiegelungen, nicht Bauteile mit gleichem Fluss (Stück 13 der Reihe).
- **Quellen:** Koopmans–Beckmann (1957) und Lawler (1963) für die beiden Formen sowie der Überblick von Burkard, Çela, Pardalos und Pitsoulis sind aus der Fachliteratur übernommen und **nicht nachgelesen**.

## Verifikation

- **Tests** (`pytest tests/`): **103 Tests**, rund anderthalb Minuten (Windows, Python 3.14). Handwerte (Handbeispiel, Distanzen, Lawler-Matrix an `n = 3`, Zählung des Flusses), Eigenschaften (Tausch-`Δ` gegen Neuberechnung, Kosten unter Rasterautomorphismen invariant, jedes globale Optimum ist ein lokales, die Einzugsgebiete decken alle `n!` Starts, der skalare Abstieg gleicht dem vektorisierten, Zahl der Optima ein Vielfaches der Automorphismen), Orakel (die Aufzählung als Referenz; `scipy.optimize.quadratic_assignment` als unabhängiger Löser, nur mit `requirements-dev.txt`), Randfälle (`n` = 2 und 3, Fluss null, vollständiger Fluss, Zwillinge, Einzelzeile, Seeds und Starts an den Grenzen, `n = 9` hinter dem Knopf), Verdikte an handgesetzten Werten, Abbildungen (feste Achsenbereiche, Datenzahl), PDF, Rauchtest und Pfade der App, Layout (Presets, jede Reglerwirkung), Footer und Laufzeit-Importe. `test_claims.py` rechnet jede Zahl dieser README nach (etwa 80 s, davon `n = 9` allein 65 s).
- **Kopie:** ein Test vergleicht die Zufallsquelle mit `lpl_rng.py` der Leiterplatte-Demo (gleiche Zahlenfolge), sofern das Schwester-Repo nebenan liegt.
- **Nicht getestet:** Zeiten; kein Mutationstest (nicht Teil des Ablaufs); Linux-CI: noch nicht gelaufen (Repo noch nicht veröffentlicht).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Grenzen, Formulierung |
| `qmh_rng.py` | SplitMix64 (Kopie aus der Leiterplatte-Demo) |
| `qmh_instance.py` | Instanz: Bauteile, Netzliste, Fluss, Raster, Startanordnungen, Handbeispiel |
| `qmh_cost.py` | Kosten, Tausch-`Δ`, Lawler-Form, Beiträge je Paar |
| `qmh_landscape.py` | Aufzählung, lokale Optima, Abstieg, Einzugsgebiete, Rasterautomorphismen |
| `qmh_evaluation.py` | Ansichten der Experimente und Verdikte |
| `qmh_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `qmh_pdf_export.py` | PDF-Export des Ergebnisses |
| `qmh_constants.py` | Größen, Raster, feste Reihen, Farben, Presets |
| `qmh_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Buttons |
| `qmh_layout.py` | Fester Kopf (Stylesheet) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Kern, Landschaft, Instanz, Randfälle, Verdikte, Abbildungen, Presets, Layout, AppTest, Orakel, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Schranken, Relaxationen, Platzierung:** die Stücke 3 bis 12 der Reihe.
- **Metaheuristiken** (Tabu, Annealing, Breakout Local Search): Fall-Demo Tor-Zuordnung und Linie Trajektorien-Metaheuristiken.
- **Andere Flussfamilien** als reglerwählbare Familie: Stück 2.
- **Symmetriereduktion der Aufzählung** (ein Bauteil auf Bahnenvertreter festlegen, Faktor 4 bis 8 schneller): ein eigener Fehlerherd, die einfache Aufzählung genügt bis `n = 9`.
- **`n = 10` live:** rund 220 s je Instanz im Prototyp; in der Vorabmessung drei Instanzen (Netzlisten-Generator mit anderem Zufallsstrom): lokale Optima je Bahn 65 bis 108, Erfolg eines Abstiegs 0,15 bis 0,48. Diese Zahlen sind **nicht** Teil der Tests.
- **Asymmetrische Flüsse und Abstände:** nur als Text (Lawler-Form).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
