"""Satellitenbeobachtungen für die Assimilation: einheitliches Format, Fehlermodell, Qualitätskontrolle,
Beobachtungsoperatoren (Plan Abschnitt 3.1, 3.2 und 5.1).

Eine Beobachtung ist eine Zeile mit
    datum, tag (0-basiert im 365-Tage-Jahr), jahr, groesse ("lai"), wert, sigma, clear, verwendet, grund
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DA, DAKonfig
from .weather import doy365


# ---------------------------------------------------------------------------
# Fehlermodell
# ---------------------------------------------------------------------------
def lai_sigma(lai, clear, rel: float = 0.25, minimum: float = 0.5):
    """Standardabweichung des S2-LAI: max(0,5; 0,25·LAI) × (1 + 2·(1 − Klar-Anteil))."""
    return np.maximum(minimum, rel * np.asarray(lai, dtype=float)) * (1.0 + 2.0 * (1.0 - np.asarray(clear, dtype=float)))


@dataclass
class Beobachtungen:
    """Beobachtungen eines Jahres als Arrays (nur die verwendeten)."""
    tag: np.ndarray
    wert: np.ndarray
    sigma: np.ndarray
    clear: np.ndarray
    groesse: str = "lai"

    def __len__(self) -> int:
        return len(self.tag)

    @property
    def lai(self) -> np.ndarray:      # Kurzform für groesse == "lai"
        return self.wert


# ---------------------------------------------------------------------------
# Beobachtungsoperatoren h(x): Modellzustand -> beobachtete Größe
# ---------------------------------------------------------------------------
def h_lai(lai_modell, cfg: DAKonfig = DA):
    """S2-LAI ≈ a · LAI_Modell + b. Standard a = 1, b = 0; a, b werden in Stufe A kalibriert,
    weil das SNAP-Netz dichtes Grünland eher unterschätzt."""
    return cfg.operator_a * np.asarray(lai_modell) + cfg.operator_b


def h_ndvi(lai_modell, a: float = 0.9, b: float = 0.75, k: float = 0.6):
    """Semi-empirischer NDVI-Operator NDVI = a − b·exp(−k·LAI) (sättigt ab LAI ≈ 3)."""
    return a - b * np.exp(-k * np.asarray(lai_modell))


# ---------------------------------------------------------------------------
# Aus dem Satelliten-Cache (heu/satellite.py) in das Beobachtungsformat
# ---------------------------------------------------------------------------
def aus_s2(s2: pd.DataFrame, schnitte: list | None = None, cfg: DAKonfig = DA) -> pd.DataFrame:
    """S2-Cache -> Beobachtungstabelle mit Qualitätskontrolle.

    schnitte: Liste von Schnittdaten (datetime.date oder Timestamp), z. B. aus detect_cuts oder vom
    Landwirt. Aufnahmen 0..sperrtage_nach_schnitt Tage danach werden nicht verwendet (Schwad, Stoppel).
    """
    cols = ["datum", "tag", "jahr", "groesse", "wert", "sigma", "clear", "verwendet", "grund"]
    if s2 is None or s2.empty or "lai" not in s2.columns:
        return pd.DataFrame(columns=cols)
    d = s2[s2["lai"].notna()].copy()
    d = d[~((d.index.month == 2) & (d.index.day == 29))]
    out = pd.DataFrame({
        "datum": d.index.normalize(),
        "tag": [doy365(t.date()) for t in d.index],
        "jahr": d.index.year,
        "groesse": "lai",
        "wert": d["lai"].to_numpy(dtype=float),
        "clear": d["clear"].to_numpy(dtype=float),
    })
    sig = lai_sigma(out["wert"], out["clear"])
    if {"lai_q25", "lai_q75"} <= set(d.columns):
        # Heterogenität im Feld als zusätzlicher Repräsentativitätsfehler (¼ der IQR)
        iqr = (d["lai_q75"] - d["lai_q25"]).to_numpy(dtype=float)
        sig = np.sqrt(sig ** 2 + np.nan_to_num(0.25 * iqr) ** 2)
    out["sigma"] = sig
    out["verwendet"] = True
    out["grund"] = ""

    def sperre(mask, grund):
        m = mask & out["verwendet"]
        out.loc[m, "verwendet"] = False
        out.loc[m, "grund"] = grund

    sperre(out["clear"] < cfg.min_clear, "bewölkt")
    month = out["datum"].dt.month
    sperre((month < 3) | (month > 11), "Winter")
    if schnitte:
        cuts = pd.to_datetime(pd.Series(list(schnitte))).dt.normalize()
        for c in cuts:
            lag = (out["datum"] - c).dt.days
            sperre((lag >= 0) & (lag <= cfg.sperrtage_nach_schnitt), "frisch gemäht")
    # Ausreißer gegen den gleitenden Median (±15 Tage) der übrigen klaren Aufnahmen desselben Abschnitts
    seg = np.zeros(len(out), dtype=int)
    if schnitte:
        for c in sorted(pd.to_datetime(list(schnitte))):
            seg += (out["datum"] > c).to_numpy().astype(int)
    # Schrittweise: jeweils die stärkste Abweichung zuerst entfernen, damit ein Ausreißer die
    # Nachbarn nicht mitreißt.
    ok = out["verwendet"].to_numpy().copy()
    vals, dates = out["wert"].to_numpy(), out["datum"].to_numpy()
    flag = np.zeros(len(out), dtype=bool)
    while True:
        worst, worst_dev = -1, 0.0
        for i in np.nonzero(ok)[0]:
            near = ok & (seg == seg[i]) & (np.abs((dates - dates[i]) / np.timedelta64(1, "D")) <= 15)
            near[i] = False
            if near.sum() >= 2:
                med = np.median(vals[near])
                dev = abs(vals[i] - med) / max(1.5, 3 * float(lai_sigma(med, 1.0)))
                if dev > 1.0 and dev > worst_dev:
                    worst, worst_dev = i, dev
        if worst < 0:
            break
        flag[worst] = True
        ok[worst] = False
    sperre(pd.Series(flag, index=out.index), "Ausreißer")
    return out[cols].reset_index(drop=True)


def jahr(tab: pd.DataFrame, jahr_: int, bis_tag: int | None = None) -> Beobachtungen:
    """Verwendete Beobachtungen eines Jahres (optional nur bis einschließlich bis_tag)."""
    t = tab[(tab["jahr"] == jahr_) & tab["verwendet"]]
    if bis_tag is not None:
        t = t[t["tag"] <= bis_tag]
    t = t.sort_values("tag").drop_duplicates("tag", keep="last")
    return Beobachtungen(t["tag"].to_numpy(dtype=int), t["wert"].to_numpy(dtype=float),
                         t["sigma"].to_numpy(dtype=float), t["clear"].to_numpy(dtype=float))


def schnitt_tage(daten: list, jahr_: int) -> list[int]:
    """Schnittdaten (date/Timestamp) eines Jahres -> 0-basierte Tage im 365-Tage-Jahr."""
    out = []
    for d in daten:
        d = pd.Timestamp(d).date()
        if d.year == jahr_ and not (d.month == 2 and d.day == 29):
            out.append(doy365(d))
    return sorted(out)
