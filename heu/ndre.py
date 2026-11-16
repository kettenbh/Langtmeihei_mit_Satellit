"""Stickstoff-Status aus dem Red-Edge (NDRE) von Sentinel-2, als Stütze für das Rohprotein.

Idee (Clevers & Gitelson 2013; Chlorophyll-Index Red-Edge):
    CIre = B8A/B05 − 1 = 2·NDRE / (1 − NDRE)
wächst etwa linear mit dem Chlorophyll je Bodenfläche (CCC = Chlorophyll je Blattfläche × LAI). Der LAI
erklärt den Großteil davon. Was bei gleichem LAI übrig bleibt, ist der Chlorophyllgehalt der Blätter, und der
folgt im Grünland eng dem N-Ernährungszustand (Chlorophyllmeter/SPAD-Studien: Blatt-N ~ Chlorophyll).

Weil es keine Futteranalysen gibt, wird nur RELATIV gerechnet, je Feld gegen seine eigene Geschichte:
    CIre_ref(LAI) = a·(1 − exp(−b·LAI)) + c      (Ausgleich über alle klaren Grünland-Aufnahmen des Feldes)
    N-Status   r  = CIre / CIre_ref(LAI)         (1 = Feld-üblich, 0,9 = 10 % weniger Chlorophyll)
Der N-Ernährungsindex des Modells (NI, aus der Feldkalibrierung = Feldmittel) wird für das Rohprotein mit
r^gamma skaliert. gamma = 1 ist eine Annahme (Chlorophyll proportional zum Blatt-N), ebenso der Rauschpegel.
Sentinel-2 LAI (SNAP-Netz) nutzt selbst die Red-Edge-Bänder; r misst also die Form des Red-Edge, nicht eine
unabhängige zweite Größe. Bei LAI < 1 dominiert der Boden, diese Aufnahmen bleiben außen vor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LAI_MIN = 1.0
B_GITTER = np.linspace(0.15, 1.5, 28)


def ci_re(ndre):
    ndre = np.asarray(ndre, float)
    return 2 * ndre / np.maximum(1 - ndre, 1e-3)


def ndre_aus_ci(ci):
    ci = np.asarray(ci, float)
    return ci / (ci + 2)


@dataclass
class Referenz:
    a: float
    b: float
    c: float
    sigma_log: float          # Streuung von log r über die Geschichte (Rauschen + echte Schwankung)
    n: int

    def __call__(self, lai):
        return self.a * (1 - np.exp(-self.b * np.asarray(lai, float))) + self.c


def _klare_gruenland(s2: pd.DataFrame) -> pd.DataFrame:
    from . import satellite as SAT
    if s2 is None or s2.empty or "ndre" not in s2 or "lai" not in s2:
        return pd.DataFrame(columns=["ndre", "lai"])
    c = s2[(s2["clear"] >= SAT.MIN_CLEAR) & s2["ndre"].notna() & s2["lai"].notna()]
    c = c[(c.index.month >= 4) & (c.index.month <= 10) & (c["lai"] >= LAI_MIN)]
    nut = SAT.nutzung_je_jahr(s2)
    return c[[nut.get(int(y)) != SAT.NUTZUNG_ACKER for y in c.index.year]]


def referenz(s2: pd.DataFrame, ohne_jahr: int | None = None) -> Referenz | None:
    """Feld-übliche Beziehung CIre(LAI). ohne_jahr: dieses Jahr nicht mitnehmen (Hindcast, kein Zirkelschluss)."""
    c = _klare_gruenland(s2)
    if ohne_jahr is not None:
        c = c[c.index.year != ohne_jahr]
    if len(c) < 15:
        return None
    x, y = c["lai"].to_numpy(float), ci_re(c["ndre"])
    best = None
    for b in B_GITTER:
        X = np.stack([1 - np.exp(-b * x), np.ones_like(x)], axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(np.sum((X @ coef - y) ** 2))
        if best is None or rss < best[0]:
            best = (rss, b, coef)
    _, b, (a, cc) = best
    ref = Referenz(float(a), float(b), float(cc), 0.0, len(c))
    lr = np.log(np.maximum(y, 1e-3) / np.maximum(ref(x), 1e-3))
    ref.sigma_log = float(np.std(lr - np.median(lr)))
    return ref


def n_status(s2: pd.DataFrame, ref: Referenz) -> pd.DataFrame:
    """Je klare Grünland-Aufnahme mit LAI ≥ 1: CIre, Referenz, r."""
    c = _klare_gruenland(s2)
    ci = ci_re(c["ndre"])
    rf = ref(c["lai"].to_numpy(float))
    return pd.DataFrame({"lai": c["lai"], "ci": ci, "ci_ref": rf, "r": ci / np.maximum(rf, 1e-3)}, index=c.index)


@dataclass
class NFaktor:
    faktor: float             # Multiplikator auf NI für das Rohprotein
    r_mittel: float           # ungeschrumpftes Mittel von r im Fenster
    n: int                    # Aufnahmen im Fenster
    von: pd.Timestamp | None
    bis: pd.Timestamp | None


def n_faktor(status: pd.DataFrame, stichtag: pd.Timestamp, letzter_schnitt: pd.Timestamp | None = None,
             fenster_tage: int = 28, sigma_aufnahme: float = 0.10, sigma_prior: float = 0.08,
             gamma: float = 1.0, grenzen=(0.7, 1.3)) -> NFaktor:
    """N-Faktor zum Stichtag aus den Aufnahmen der letzten `fenster_tage` (frühestens ab dem letzten Schnitt;
    direkt nach dem Schnitt sind die Stoppeln nicht repräsentativ, darum erst 7 Tage danach).
    Bayes-Schrumpfung: log r ~ N(0, sigma_prior) a priori, je Aufnahme Rauschen sigma_aufnahme.
    Standardwerte aus Niederham 2017–2026 (153 Aufnahmen): Streuung von log r 0,13, davon Rauschen ≈ 0,10
    (aus Paaren von Aufnahmen ≤ 10 Tage auseinander), bleibt ein echtes Signal von ≈ 0,08 (±8 % N-Status)."""
    t0 = stichtag - pd.Timedelta(days=fenster_tage)
    if letzter_schnitt is not None:
        t0 = max(t0, letzter_schnitt + pd.Timedelta(days=7))
    w = status[(status.index > t0) & (status.index <= stichtag)]
    if w.empty:
        return NFaktor(1.0, float("nan"), 0, None, None)
    lr = np.log(w["r"].to_numpy(float))
    prec = len(lr) / sigma_aufnahme ** 2 + 1 / sigma_prior ** 2
    post = (lr.sum() / sigma_aufnahme ** 2) / prec
    f = float(np.clip(np.exp(gamma * post), *grenzen))
    return NFaktor(f, float(np.exp(lr.mean())), len(lr), w.index.min(), w.index.max())
