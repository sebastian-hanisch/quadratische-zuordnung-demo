"""Konstanten: Größen, Raster, feste Reihen der Experimente, Farben und Presets."""

SIZES = (4, 6, 8, 9)                                  # Bauteile = Plätze (alle n! Anordnungen lassen sich aufzählen)
DEFAULT_N = 8
GRID = {4: (2, 2), 6: (2, 3), 8: (2, 4), 9: (3, 3)}  # Plätze als Raster (Zeilen, Spalten), Abstand = Manhattan
SEED_MAX, DEFAULT_SEED = 999999, 35
START_MAX, DEFAULT_START = 999999, 1                  # „Zufallsstart Nr.“ für Tausch und Abstieg
COMPONENT_CHOICES = tuple(range(1, 10))               # Bauteil a, b (1-basiert, begrenzt auf n)
DEFAULT_A, DEFAULT_B = 1, 2
N_NETS_PER_COMPONENT = 1.5                            # Zahl der Netze = gerundet 1,5 n (mindestens 2)
NET_SIZES = (2, 3, 4)
NET_WEIGHTS = (5, 3, 2)                               # Gewichte der Netzgrößen (ganzzahlig, für SplitMix64.weighted)
LOCAL_NET_PERCENT = 70                                # Anteil der Netze innerhalb einer Dreiergruppe (IC mit Kondensator und Widerstand)
GROUP = 3

SWEEP_SIZES = (4, 6, 8)
SWEEP_SEEDS = tuple(range(100000, 100020))            # feste Instanzen: nie ein einzelner Seed als Befund
SWEEP_SEEDS_9 = SWEEP_SEEDS[:5]                       # n = 9 braucht je Instanz rund 13 s: nur fünf feste Instanzen
HAND_FLOWS = ((0, 1, 3), (0, 2, 1), (1, 3, 2), (2, 3, 4))   # Handbeispiel n = 4: Fluss zwischen den Bauteilen (a, b, Menge)
HAND_GRID = (2, 2)

COLORS = {"best": "#2F6B65", "local": "#c0504d", "other": "rgba(150,150,150,0.6)", "line": "#4c78a8", "net": "#888888", "ok": "#2F6B65", "warn": "#b08a2e", "bad": "#b34a4a"}

_BASE = {"n": DEFAULT_N, "seed": DEFAULT_SEED, "a": DEFAULT_A, "b": DEFAULT_B, "start": DEFAULT_START}
PRESETS = {
    "Standardfall: acht Bauteile": dict(_BASE),
    "Kleinste Halle: vier Bauteile": dict(_BASE, n=4, b=3),
    "Sechs Bauteile: alle 720 Anordnungen": dict(_BASE, n=6),
    "Neun Bauteile: 362 880 Anordnungen": dict(_BASE, n=9, seed=100000),
    "Anderer Start (Nr. 7)": dict(_BASE, start=7),
    "Tausch zweier Bauteile (3 und 6)": dict(_BASE, a=3, b=6),
}
PRESET_HELP = {
    "Standardfall: acht Bauteile": "Acht Bauteile auf einem Raster aus zwei Reihen mit vier Plätzen: 40 320 Anordnungen, alle werden gerechnet. Die Netzliste ergibt die Flüsse zwischen den Bauteilen.",
    "Kleinste Halle: vier Bauteile": "Vier Bauteile auf vier Plätzen: 24 Anordnungen. Das Raster hat Spiegelungen und Drehungen, deshalb gibt es mehrere gleich gute Anordnungen.",
    "Sechs Bauteile: alle 720 Anordnungen": "Sechs Bauteile: 720 Anordnungen, 15 mögliche Tausche. Die Landschaft ist klein genug, um jede lokale Spitze einzeln zu betrachten.",
    "Neun Bauteile: 362 880 Anordnungen": "Neun Bauteile auf einem 3×3-Raster: 362 880 Anordnungen. Die Rechnung läuft auf Knopfdruck (rund eine Viertelminute).",
    "Anderer Start (Nr. 7)": "Ein anderer Zufallsstart für Tausch und Abstieg: der Abstieg endet nicht unbedingt im selben lokalen Optimum.",
    "Tausch zweier Bauteile (3 und 6)": "Bauteil 3 und Bauteil 6 tauschen die Plätze: die Änderung der Kosten nach der Formel und nach der Neuberechnung.",
}


def n_text(n: int) -> str:
    """Anzahl mit Leerzeichen als Tausendertrenner: 362880 → „362 880“."""
    return f"{n:,}".replace(",", " ")


assert set(PRESETS) == set(PRESET_HELP) and all(p["n"] in SIZES and 1 <= p["a"] <= 9 and 1 <= p["b"] <= 9 and 0 <= p["seed"] <= SEED_MAX and 1 <= p["start"] <= START_MAX for p in PRESETS.values())
assert all(r * c == n for n, (r, c) in GRID.items())
