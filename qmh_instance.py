"""Instanz: Bauteile, Netzliste, Fluss und Abstände der Plätze (Raster, Manhattan). Alle Zufallsentscheidungen über SplitMix64.

Jedes Paar von Bauteilen in einem Netz zählt einen Fluss von 1 (Clique, ganzzahlig); die Flussmatrix F ist symmetrisch mit Diagonale 0. Die Plätze liegen auf einem Raster, D ist der Manhattan-Abstand."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

import qmh_constants as C
from qmh_rng import SplitMix64


@dataclass(frozen=True, eq=False)
class Instance:
    n: int
    rows: int
    cols: int
    F: np.ndarray            # (n, n) ganzzahlig, symmetrisch, Diagonale 0
    D: np.ndarray            # (n, n) Abstand der Plätze
    names: tuple             # Bauteilnamen (IC1, C1, R1, IC2, ...)
    nets: tuple              # je Netz ein Tupel von Bauteilnummern (0-basiert)
    seed: int


def grid_distance(rows: int, cols: int) -> np.ndarray:
    """Manhattan-Abstand zwischen den Plätzen eines Rasters; Platz = Zeile · Spalten + Spalte."""
    if rows < 1 or cols < 1:
        raise ValueError("Raster braucht mindestens eine Zeile und eine Spalte")
    xy = np.array([(r, c) for r in range(rows) for c in range(cols)])
    return np.abs(xy[:, None, :] - xy[None, :, :]).sum(axis=2).astype(np.int16)


def component_names(n: int) -> tuple:
    """IC1, C1, R1, IC2, C2, R2, ...: je drei Bauteile eine Gruppe (ein IC mit Kondensator und Widerstand)."""
    kinds = ("IC", "C", "R")
    return tuple(f"{kinds[i % 3]}{i // 3 + 1}" for i in range(n))


def flow_from_nets(n: int, nets) -> np.ndarray:
    F = np.zeros((n, n), dtype=np.int16)
    for net in nets:
        for a, b in combinations(net, 2):
            F[a, b] += 1
            F[b, a] += 1
    return F


def make_instance(n: int, seed: int) -> Instance:
    if n not in C.GRID:
        raise ValueError(f"n muss eine der Größen {C.SIZES} sein")
    rng = SplitMix64(seed * 1000003 + n)
    groups = (n + C.GROUP - 1) // C.GROUP
    nets = []
    for _ in range(max(2, int(round(C.N_NETS_PER_COMPONENT * n)))):
        k = min(rng.weighted(C.NET_SIZES, C.NET_WEIGHTS), n)
        pool = list(range(n))
        if rng.below(100) < C.LOCAL_NET_PERCENT:
            g = rng.below(groups)
            local = [i for i in range(n) if i // C.GROUP == g]
            if len(local) >= k:
                pool = local
        rng.shuffle(pool)
        nets.append(tuple(sorted(pool[:k])))
    rows, cols = C.GRID[n]
    return Instance(n, rows, cols, flow_from_nets(n, nets), grid_distance(rows, cols), component_names(n), tuple(nets), seed)


def hand_instance() -> Instance:
    """Das Handbeispiel n = 4 (Raster 2×2): Fluss 0–1: 3, 0–2: 1, 1–3: 2, 2–3: 4; der Rechenweg steht in der README."""
    n = 4
    F = np.zeros((n, n), dtype=np.int16)
    for a, b, w in C.HAND_FLOWS:
        F[a, b] = F[b, a] = w
    rows, cols = C.HAND_GRID
    return Instance(n, rows, cols, F, grid_distance(rows, cols), ("A", "B", "C", "D"), (), 0)


def random_start(n: int, start: int) -> list:
    """Die Startanordnung Nr. `start` (Fisher-Yates mit SplitMix64): p[i] = Platz des Bauteils i."""
    rng = SplitMix64(start * 7919 + n)
    p = list(range(n))
    rng.shuffle(p)
    return p
