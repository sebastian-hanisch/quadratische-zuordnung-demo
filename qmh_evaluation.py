"""Kennzahlen, Experimente und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text).

Alle Aussagen gelten für die gerechnete Instanz und die Familie „Netzliste“ (siehe README); Zeiten stehen nie in Verdikten."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations

import numpy as np

import qmh_constants as C
import qmh_cost as K
import qmh_landscape as L
from qmh_instance import hand_instance, make_instance, random_start


@dataclass(frozen=True)
class Settings:
    seed: int = C.DEFAULT_SEED
    n: int = C.DEFAULT_N
    a: int = C.DEFAULT_A
    b: int = C.DEFAULT_B
    start: int = C.DEFAULT_START


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def pct(x: float, digits: int = 0) -> str:
    return de(100.0 * x, digits) + " %"


def pair_of(s: Settings) -> tuple:
    """Die gewählten Bauteile a und b (0-basiert), begrenzt auf n; sind beide gleich, rückt b um eins weiter."""
    a = min(s.a, s.n) - 1
    b = min(s.b, s.n) - 1
    if a == b:
        b = (a + 1) % s.n
    return a, b


def landscape_of(s: Settings) -> L.Landscape:
    return L.landscape(make_instance(s.n, s.seed))


# ----------------------------------------------------------------------------- Ergebnis: die beste Anordnung

def board_view(s: Settings, ls: L.Landscape) -> dict:
    inst = make_instance(s.n, s.seed)
    best = [int(x) for x in ls.optima[0]]
    return {"inst": inst, "best": best, "cost": ls.opt, "pairs": K.pair_costs(best, inst.F, inst.D), "worst": int(ls.cost_values[-1]), "mean": ls.mean_cost, "n_opt": ls.n_opt, "aut": ls.aut}


def hand_view() -> dict:
    """Das Handbeispiel n = 4: alle 24 Anordnungen mit Kosten (von Hand prüfbar, siehe README)."""
    inst = hand_instance()
    rows = sorted((K.cost(p, inst.F, inst.D), p) for p in permutations(range(4)))
    values: dict = {}
    for c, _ in rows:
        values[c] = values.get(c, 0) + 1
    return {"inst": inst, "rows": rows, "values": values}


# ----------------------------------------------------------------------------- Experiment 2: alle Anordnungen

def landscape_view(s: Settings, ls: L.Landscape) -> dict:
    nonglobal = [i for i in range(ls.n_local) if ls.local_cost[i] != ls.opt]
    return {"ls": ls, "n_local": ls.n_local, "strict": ls.n_local_strict, "local_orbits": ls.local_orbits, "optima_orbits": ls.optima_orbits, "n_nonglobal": len(nonglobal),
            "share_local": ls.n_local / ls.perms, "above_opt": (ls.mean_cost - ls.opt) / ls.opt if ls.opt else float("nan")}


# ----------------------------------------------------------------------------- Experiment 3: der Tausch

def swap_view(s: Settings, ls: L.Landscape) -> dict:
    inst = make_instance(s.n, s.seed)
    p = random_start(s.n, s.start)
    a, b = pair_of(s)
    q = K.swapped(p, a, b)
    c_p = K.cost(p, inst.F, inst.D)
    c_q = K.cost(q, inst.F, inst.D)
    deltas = [(x, y, K.swap_delta(p, x, y, inst.F, inst.D)) for x, y in K.pair_list(s.n)]
    best = min(deltas, key=lambda t: (t[2], K.pair_list(s.n).index((t[0], t[1]))))
    # ein lokales Optimum, das kein globales ist (das mit dem größten Einzugsgebiet)
    cand = [i for i in range(ls.n_local) if ls.local_cost[i] != ls.opt]
    example = None
    if cand:
        i = max(cand, key=lambda j: (int(ls.local_basin[j]), -j))
        example = {"p": [int(x) for x in ls.local[i]], "cost": int(ls.local_cost[i]), "basin": int(ls.local_basin[i]), "strict": bool(ls.local_strict[i])}
    return {"inst": inst, "p": p, "q": q, "a": a, "b": b, "cost_p": c_p, "cost_q": c_q, "delta_formula": K.swap_delta(p, a, b, inst.F, inst.D), "delta_recomputed": c_q - c_p,
            "deltas": deltas, "best": best, "improving": sum(1 for t in deltas if t[2] < 0), "is_local": all(t[2] >= 0 for t in deltas), "example": example}


# ----------------------------------------------------------------------------- Experiment 4: ein Abstieg und viele Starts

def descent_view(s: Settings, ls: L.Landscape) -> dict:
    inst = make_instance(s.n, s.seed)
    p = random_start(s.n, s.start)
    path = L.descent(inst, p)
    costs = [K.cost(x, inst.F, inst.D) for x in path]
    end = path[-1]
    return {"inst": inst, "path": path, "costs": costs, "end": end, "end_cost": costs[-1], "steps": len(path) - 1, "global": costs[-1] == ls.opt, "basin": L.basin_of(ls, end), "success": ls.success,
            "restarts": ls.restarts, "perms": ls.perms}


# ----------------------------------------------------------------------------- Experiment 5: Wachstum mit n

def growth_rows(sizes=C.SWEEP_SIZES, seeds=C.SWEEP_SEEDS, extra=None) -> list:
    """Je Größe die Landschaften der festen Instanzen: Median der lokalen Optima je Bahn und des Erfolgs eines Starts. `extra`: {n: seeds} für weitere Größen mit anderen Seeds."""
    plan = [(n, tuple(seeds)) for n in sizes] + [(n, tuple(sd)) for n, sd in (extra or {}).items()]
    rows = []
    for n, sd in plan:
        lss = [L.landscape(make_instance(n, x)) for x in sd]
        rows.append({"n": n, "perms": lss[0].perms, "seeds": len(sd), "local_orbits": float(np.median([x.local_orbits for x in lss])), "local_min": min(x.local_orbits for x in lss),
                     "local_max": max(x.local_orbits for x in lss), "success": float(np.median([x.success for x in lss])), "success_min": min(x.success for x in lss),
                     "success_max": max(x.success for x in lss), "steps": float(np.mean([x.mean_steps for x in lss])), "opt_orbits": float(np.median([x.optima_orbits for x in lss]))})
    return rows


# ----------------------------------------------------------------------------- Verdikte

def verdict_board(s: Settings, v: dict, ls: L.Landscape):
    orbits = ls.optima_orbits
    bahn = "eine Bahn" if orbits == 1 else f"{de(orbits, 0) if orbits == int(orbits) else de(orbits, 1)} Bahnen"
    same = (f"{v['n_opt']} Anordnungen erreichen diesen Wert, das sind {bahn} unter den {ls.aut} Spiegelungen und Drehungen des Rasters" if v["n_opt"] > 1 else "genau eine Anordnung erreicht diesen Wert")
    return "success", (f"Die beste von {C.n_text(ls.perms)} Anordnungen kostet **{v['cost']}**, die schlechteste {v['worst']}, im Mittel kosten sie {de(v['mean'], 1)}; {same}. "
                       f"Alles ist aufgezählt: das Optimum ist bewiesen, nicht gesucht.")


def verdict_landscape(s: Settings, v: dict):
    ls = v["ls"]
    kind = "warning" if v["n_nonglobal"] > 0 else "success"
    return kind, (f"Von den {C.n_text(ls.perms)} Anordnungen sind **{C.n_text(ls.n_local)}** lokale Optima der Tauschnachbarschaft ({de(v['local_orbits'], 1)} je Bahn), davon {ls.n_local_strict} streng (jeder Tausch macht es schlechter); "
                  f"nur {ls.n_opt} sind global optimal. {v['n_nonglobal']} lokale Optima sind kein globales Optimum: dort bleibt eine Tauschsuche stehen. Gezählt sind schwache Optima (Gleichstände erlaubt); "
                  f"Spiegelbilder haben dieselben Kosten, deshalb zählt die Demo auch je Bahn.")


def verdict_swap(s: Settings, v: dict):
    names = v["inst"].names
    a, b = names[v["a"]], names[v["b"]]
    ok = v["delta_formula"] == v["delta_recomputed"]
    sign = "senkt" if v["delta_formula"] < 0 else ("erhöht" if v["delta_formula"] > 0 else "ändert nicht")
    tail = f" Von den {len(v['deltas'])} möglichen Tauschen verbessern {v['improving']} die Startanordnung." if not v["is_local"] else f" Keiner der {len(v['deltas'])} möglichen Tausche verbessert sie: die Startanordnung ist ein lokales Optimum."
    sd = lambda x: "0" if x == 0 else f"{x:+d}".replace("-", "−")
    return ("success" if ok else "error"), (f"Der Tausch von {a} und {b} {sign} die Kosten von {v['cost_p']} auf {v['cost_q']} (Δ = {sd(v['delta_formula'])}); die Formel über die anderen n − 2 Bauteile "
                                           f"{'stimmt mit der Neuberechnung überein' if ok else 'WEICHT von der Neuberechnung ab'} (Δ = {sd(v['delta_recomputed'])}).{tail}")


def verdict_descent(s: Settings, v: dict):
    where = "ein **globales Optimum**" if v["global"] else "ein **lokales Optimum, das kein globales ist**"
    return ("success" if v["global"] else "warning"), (f"Der steilste Abstieg von Start Nr. {s.start} endet nach {v['steps']} Tauschen bei Kosten {v['end_cost']}: {where}. Von allen {C.n_text(v['perms'])} Startanordnungen erreichen "
                                                      f"**{pct(v['success'], 1)}** ein globales Optimum; im Mittel brauchte man also {de(v['restarts'], 1)} Zufallsstarts bis zum ersten Treffer. "
                                                      f"Im Einzugsgebiet dieses Endpunkts enden {C.n_text(v['basin'])} Starts.")


def verdict_growth(rows: list):
    first, last = rows[0], rows[-1]
    grows = all(rows[i + 1]["local_orbits"] > rows[i]["local_orbits"] for i in range(len(rows) - 1))
    falls = all(rows[i + 1]["success"] < rows[i]["success"] for i in range(len(rows) - 1))
    parts = "; ".join(f"n = {r['n']}: {de(r['local_orbits'], 0)} lokale Optima je Bahn, Erfolg eines Starts {pct(r['success'])} ({r['seeds']} Instanzen)" for r in rows)
    claim = ("Im Median wächst die Zahl der lokalen Optima mit n und der Erfolg eines einzelnen Abstiegs sinkt" if grows and falls else
             "Die Zahl der lokalen Optima und der Erfolg eines Abstiegs verlaufen in dieser Messung nicht gleichmäßig")
    return ("success" if grows and falls else "info"), (f"{claim} (Median über feste Instanzen, Familie Netzliste; die Spannen der Instanzen überlappen zwischen den Größen): {parts}. Die Zahl der Anordnungen wächst von {C.n_text(first['perms'])} auf {C.n_text(last['perms'])}.")


def verdict_core(s: Settings, v: dict, ls: L.Landscape, d: dict):
    return "info", (f"Bei {s.n} Bauteilen lässt sich alles aufzählen, und die Landschaft zeigt, was die Tauschsuche kann: {C.n_text(ls.n_local)} von {C.n_text(ls.perms)} Anordnungen sind lokale Optima, "
                    f"ein einzelner Abstieg erreicht in {pct(ls.success)} der Starts ein globales Optimum. Schon bei zwölf Bauteilen sind es {C.n_text(math.factorial(12))} Anordnungen: dann ist das Aufzählen unpraktisch, und die Landschaft bleibt unsichtbar.")
