"""Modul 1: Hitzestress der Milchkühe über den Temperatur-Feuchte-Index (THI).

THI nach NRC (1971):        THI = (1,8·T + 32) − (0,55 − 0,0055·RH)·(1,8·T − 26)
Weide-THI nach Mader et al. (2006):  THI_adj = 4,51 + THI − 1,992·WS + 0,0068·RAD   (WS m/s, RAD W/m²)
Schwellen: 68 Beginn (Zimbelman et al. 2009), 72/80/90 leicht/mittel/schwer (Armstrong 1994).
Milchverlust: linear je THI-Punkt des Tagesmittels über 68 (Größenordnung West 2003, Bouraoui et al. 2002).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HitzeAnnahmen

STUFEN = ["keine", "Beginn", "leicht", "mittel", "schwer"]
STUFEN_FARBE = ["#2e9d5b", "#c9b52b", "#e8892b", "#d9534f", "#7b1f1f"]
HINWEISE = [
    "Keine Maßnahmen nötig.",
    "Tränken kontrollieren, Weidegang in die kühleren Stunden legen.",
    "Weide nur nachts/früh morgens, Stall-Lüfter an, zusätzliche Tränken, Futter frisch vorlegen.",
    "Tiere im Stall mit Lüftern/Sprühkühlung halten, Fütterung in die Abend- und Nachtstunden, Herde beobachten.",
    "Notfallmaßnahmen: maximale Kühlung, Tierarzt informieren, keine Treib- oder Behandlungsarbeiten.",
]


def thi(t, rh):
    t = np.asarray(t, float)
    rh = np.asarray(rh, float)
    return (1.8 * t + 32) - (0.55 - 0.0055 * rh) * (1.8 * t - 26)


def thi_adj(thi_val, ws, rad):
    return np.asarray(thi_val, float) + 4.51 - 1.992 * np.asarray(ws, float) + 0.0068 * np.asarray(rad, float)


def mit_thi(hourly: pd.DataFrame) -> pd.DataFrame:
    d = hourly.copy()
    d["thi"] = thi(d["t"], d["rh"])
    d["thi_weide"] = thi_adj(d["thi"], d["ws"], d["rad"])
    return d


def _stufe(thi_max: np.ndarray, h: HitzeAnnahmen) -> np.ndarray:
    return np.select([thi_max >= h.schwelle_schwer, thi_max >= h.schwelle_mittel, thi_max >= h.schwelle_leicht,
                      thi_max >= h.schwelle_beginn], [4, 3, 2, 1], 0)


def tageskennzahlen(hourly: pd.DataFrame, h: HitzeAnnahmen, spalte: str = "thi") -> pd.DataFrame:
    """Tageswerte: thi_max, thi_mittel, stunden_68, last_68 (Σ THI−68), erholung_nacht, stufe, milchverlust."""
    d = hourly if "thi" in hourly else mit_thi(hourly)
    x = d[spalte]
    g = x.groupby(d.index.normalize())
    nacht = x[(d.index.hour >= 22) | (d.index.hour < 6)]
    erh = (nacht < h.erholung_thi).groupby(nacht.index.normalize()).sum()
    out = pd.DataFrame({
        "thi_max": g.max(), "thi_mittel": g.mean(),
        "stunden_68": (x >= h.schwelle_beginn).groupby(d.index.normalize()).sum(),
        "last_68": (x - h.schwelle_beginn).clip(lower=0).groupby(d.index.normalize()).sum(),
    })
    out["erholung_nacht"] = (erh.reindex(out.index).fillna(0) >= h.erholung_stunden)
    stufe = _stufe(out["thi_max"].to_numpy(), h)
    # Mehrtageseffekt: 3 Tage in Folge ≥ 8 Hitzestunden ohne Nachterholung -> eine Stufe höher
    belastet = ((out["stunden_68"] >= 8) & ~out["erholung_nacht"]).astype(int)
    serie = belastet.rolling(3, min_periods=3).sum().fillna(0).to_numpy() >= 3
    stufe = np.where(serie & (stufe > 0), np.minimum(stufe + 1, 4), stufe)
    out["stufe"] = stufe
    out["milchverlust_kg"] = (out["thi_mittel"] - h.schwelle_beginn).clip(lower=0) * h.milchverlust_kg_je_thi
    return out


def ensemble_tage(ens: pd.DataFrame, h: HitzeAnnahmen, spalte: str = "thi") -> pd.DataFrame:
    """Je Member Tageskennzahlen, dann je Tag Wahrscheinlichkeiten der Stufen."""
    ens = mit_thi(ens)
    rows = []
    for m, d in ens.groupby("member"):
        t = tageskennzahlen(d.drop(columns="member"), h, spalte)
        t["member"] = m
        rows.append(t)
    allm = pd.concat(rows)
    g = allm.groupby(level=0)
    res = pd.DataFrame({
        "n": g.size(),
        "thi_max_q10": g["thi_max"].quantile(0.1), "thi_max_q50": g["thi_max"].median(),
        "thi_max_q90": g["thi_max"].quantile(0.9),
        "stunden_68_q50": g["stunden_68"].median(),
        "milchverlust_q50": g["milchverlust_kg"].median(),
    })
    for k in range(1, 5):
        res[f"p_stufe{k}"] = g["stufe"].apply(lambda s, k=k: float(np.mean(s >= k)))
    res["stufe_wahrscheinlich"] = g["stufe"].median().round().astype(int)
    return res


def ensemble_stunden(ens: pd.DataFrame, spalte: str = "thi") -> pd.DataFrame:
    e = mit_thi(ens)
    g = e.groupby(level=0)[spalte]
    return pd.DataFrame({q: g.quantile(q / 100) for q in (10, 25, 50, 75, 90)})


def historie(hourly: pd.DataFrame, h: HitzeAnnahmen, von_monat: int = 5, bis_monat: int = 9) -> pd.DataFrame:
    """Je Jahr und Tag (1.5.–30.9.): kumulierte Hitzestunden (THI ≥ 68). Zeilen = Jahre, Spalten = Tag der Saison."""
    t = tageskennzahlen(hourly, h)
    t = t[(t.index.month >= von_monat) & (t.index.month <= bis_monat)]
    t = t[~((t.index.month == 2) & (t.index.day == 29))]
    t["jahr"] = t.index.year
    t["tag"] = t.groupby("jahr").cumcount()
    return t.pivot(index="jahr", columns="tag", values="stunden_68").cumsum(axis=1)


def jahreswerte(hourly: pd.DataFrame, h: HitzeAnnahmen) -> pd.DataFrame:
    t = tageskennzahlen(hourly, h)
    t = t[(t.index.month >= 5) & (t.index.month <= 9)]
    g = t.groupby(t.index.year)
    return pd.DataFrame({"hitzestunden_68": g["stunden_68"].sum(), "tage_thi72": g["thi_max"].apply(lambda s: int((s >= 72).sum())),
                         "milchverlust_kg_kuh": g["milchverlust_kg"].sum()})
