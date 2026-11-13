"""Zwillingsexperiment (OSSE) für die Assimilation, ohne Internet (Plan Abschnitt 4).

Idee: Ein "wahres" Feld wird mit bekannten, absichtlich anderen Parametern simuliert. Daraus entstehen
synthetische Sentinel-2-LAI-Beobachtungen mit Wolkenlücken und Messfehler. Ein Prior-Ensemble mit den
Standardannahmen der App läuft dann gegen diese Wahrheit.

Paket 1 liefert die Bausteine und den Open-Loop-Lauf (Ensemble ohne Korrektur) als Messlatte.
Paket 3 setzt den EnKF-Schritt an den Beobachtungstagen dazwischen und muss besser abschneiden.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import ensemble as E
from .config import Annahmen
from .model import ModVegeParams


# ---------------------------------------------------------------------------
# Beobachtungsfehler (Plan 3.2)
# ---------------------------------------------------------------------------
def lai_sigma(lai, clear, rel: float = 0.25, minimum: float = 0.5):
    """Standardabweichung des S2-LAI: max(0,5; 0,25·LAI) × (1 + 2·(1 − Klar-Anteil))."""
    return np.maximum(minimum, rel * np.asarray(lai)) * (1.0 + 2.0 * (1.0 - np.asarray(clear)))


@dataclass
class Beobachtungen:
    tag: np.ndarray        # 0-basierter Tag im 365-Tage-Jahr
    lai: np.ndarray        # beobachteter LAI
    sigma: np.ndarray      # angenommener Beobachtungsfehler
    clear: np.ndarray      # Anteil wolkenfreier Feldpixel


def synthetische_lai(lai_wahr: np.ndarray, rng: np.random.Generator, revisit: int = 5,
                     p_bewoelkt: float = 0.55, min_clear: float = 0.8, bias: float = 0.0,
                     start: int = 60, ende: int = 300) -> Beobachtungen:
    """S2-ähnliche LAI-Reihe aus einer wahren LAI-Reihe: Überflug alle `revisit` Tage,
    mit Wahrscheinlichkeit p_bewoelkt unbrauchbar; brauchbare Aufnahmen mit Klar-Anteil 0,8–1,0
    und Gauß-Fehler nach lai_sigma. `bias` simuliert einen systematischen Fehler des LAI-Produkts."""
    tage, werte, sig, clr = [], [], [], []
    for d in range(start, ende, revisit):
        if rng.random() < p_bewoelkt:
            continue
        c = rng.uniform(min_clear, 1.0)
        s = float(lai_sigma(lai_wahr[d], c))
        tage.append(d)
        werte.append(max(0.0, lai_wahr[d] + bias + rng.normal(0.0, s)))
        sig.append(s)
        clr.append(c)
    return Beobachtungen(np.array(tage, dtype=int), np.array(werte), np.array(sig), np.array(clr))


# ---------------------------------------------------------------------------
# Wahrheit und Prior
# ---------------------------------------------------------------------------
@dataclass
class Wahrheit:
    p: ModVegeParams
    zustand: E.EnsembleState     # n = 1
    trace: dict                  # Tageswerte (365, 1)
    schnitt_tage: list


def wahrheit(w: dict, p_wahr: ModVegeParams, a: Annahmen) -> Wahrheit:
    s, tr = E.run_ensemble(w, E.EnsembleParams.from_params(p_wahr, 1), a, trace=True)
    tage = [int(d) for d in s.cut_day[0] if d >= 0]
    return Wahrheit(p_wahr, s, tr, tage)


PRIOR_SIGMA = {"NI": 0.2, "RUEmax": 0.1, "WHC": 0.2}


def prior_ensemble(p: ModVegeParams, n: int, rng: np.random.Generator,
                   log_sigma: dict | None = None, zustand_sigma: float = 0.3, a: Annahmen | None = None):
    """Prior-Ensemble: gestörte Parameter um p und gestörter Startzustand."""
    ep = E.perturb_params(p, n, rng, log_sigma or PRIOR_SIGMA)
    s0 = E.EnsembleState.initial(ep, max_cuts=(a.max_schnitte + 2) if a else 8)
    return ep, E.perturb_state(s0, rng, zustand_sigma)


def schnitt_ensemble(tage: list, n: int, rng: np.random.Generator, unsicherheit_tage: int = 0) -> np.ndarray:
    """Beobachtete Schnitttage je Mitglied, optional gleichverteilt um ±unsicherheit_tage verschoben
    (Plan 3.4: Schnitt aus Satellit nur auf ein Zeitfenster genau bekannt)."""
    t = np.array(tage, dtype=int)[None, :].repeat(n, axis=0)
    if unsicherheit_tage:
        t = t + rng.integers(-unsicherheit_tage, unsicherheit_tage + 1, size=t.shape)
    return t


# ---------------------------------------------------------------------------
# Kennzahlen
# ---------------------------------------------------------------------------
def crps_ensemble(x: np.ndarray, y: float) -> float:
    """CRPS einer Ensemble-Stichprobe x gegen den wahren Wert y (kleiner = besser)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    term1 = np.mean(np.abs(x - y))
    term2 = np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * n)
    return float(term1 - term2)


def bewertung(lai_ens: np.ndarray, lai_wahr: np.ndarray, ertrag_ens: np.ndarray, ertrag_wahr: float,
              tage: slice | np.ndarray) -> dict:
    """LAI-Fehler (Ensemblemittel gegen Wahrheit), mittlere Streuung, Spread/Skill, Ertrags-CRPS und Rang."""
    m = lai_ens[tage].mean(axis=1)
    rmse = float(np.sqrt(np.mean((m - lai_wahr[tage]) ** 2)))
    spread = float(np.sqrt(np.mean(lai_ens[tage].var(axis=1, ddof=1))))
    return {
        "lai_rmse": rmse,
        "lai_spread": spread,
        "spread_skill": spread / rmse if rmse > 0 else np.nan,
        "ertrag_mittel": float(np.mean(ertrag_ens)),
        "ertrag_wahr": float(ertrag_wahr),
        "ertrag_fehler": float(np.mean(ertrag_ens) - ertrag_wahr),
        "ertrag_crps": crps_ensemble(ertrag_ens, ertrag_wahr),
        "rang_wahrheit": float(np.mean(ertrag_ens < ertrag_wahr)),
    }


def open_loop(w: dict, wahr: Wahrheit, p_prior: ModVegeParams, a: Annahmen, n: int = 64, seed: int = 0,
              schnitt_unsicherheit: int = 3) -> tuple[E.EnsembleState, dict]:
    """Prior-Ensemble mit wahrem Wetter und beobachteten Schnitten, ohne Assimilation."""
    rng = np.random.default_rng(seed)
    ep, s0 = prior_ensemble(p_prior, n, rng, a=a)
    fc = schnitt_ensemble(wahr.schnitt_tage, n, rng, schnitt_unsicherheit)
    return E.run_ensemble(w, ep, a, state=s0, manage=False, forced_cuts=fc, trace=True)


def wahre_parameter(p: ModVegeParams, **abweichung) -> ModVegeParams:
    return replace(p, **abweichung)
