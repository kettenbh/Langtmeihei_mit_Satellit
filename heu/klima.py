"""Modul 4: Klimatrend für den Standort (Kennzahlen je Jahr, Trends).

Kennzahlen
- Vegetationsbeginn: Grünlandtemperatursumme GTS = Σ positiver Tagesmittel ab 1.1., Januar ×0,5, Februar ×0,75;
  Beginn = Tag, an dem GTS 200 erreicht.
- Thermische Vegetationsperiode: Beginn = erster Tag der ersten Folge von 6 Tagen mit Tmittel > 5 °C,
  Ende = erster Tag der ersten Folge von 6 Tagen mit Tmittel < 5 °C nach dem 1. Juli.
- Sommertage (Tmax ≥ 25), Hitzetage (Tmax ≥ 30), Tropennächte (Tmin ≥ 20), Frosttage (Tmin < 0),
  letzter Frost nach Vegetationsbeginn.
- Trockenphasen Apr–Sep: längste Folge von Tagen < 1 mm, Zahl der Folgen ≥ 14 Tage; KWB Apr–Sep = Σ(N − ET0).
- SPEI-3 (Vicente-Serrano et al. 2010), hier nichtparametrisch (Rang -> Normalquantil, Gringorten-Formel)
  statt log-logistischer Anpassung; Anteil Monate Apr–Sep mit SPEI < −1.
- Heuwetterfenster Mai–Sep: nicht überlappende Fenster mit der Heuwetter-Regel der App (model.dry_window).
- Schnittzahl, 1. Schnitttermin und Wasserstresstage aus der ModVege-Simulation.
Trend: Mann-Kendall-Test (Kendalls τ gegen Jahr) und Theil-Sen-Steigung mit 90-%-Vertrauensbereich.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _runs(mask: np.ndarray, laenge: int) -> int | None:
    """Index des ersten Tages der ersten Folge von 'laenge' True-Werten."""
    c = 0
    for i, v in enumerate(mask):
        c = c + 1 if v else 0
        if c >= laenge:
            return i - laenge + 1
    return None


def _laengste_folge(mask: np.ndarray) -> tuple[int, int]:
    best = c = n14 = 0
    for v in list(mask) + [False]:
        if v:
            c += 1
        else:
            if c >= 14:
                n14 += 1
            best = max(best, c)
            c = 0
    return best, n14


def gts_tag(d: pd.DataFrame, ziel: float = 200.0) -> float:
    t = d["tmean"].clip(lower=0).to_numpy()
    f = np.where(d.index.month == 1, 0.5, np.where(d.index.month == 2, 0.75, 1.0))
    c = np.cumsum(t * f)
    i = np.argmax(c >= ziel)
    return float(d.index[i].dayofyear) if c[-1] >= ziel else np.nan


def spei3(df: pd.DataFrame) -> pd.Series:
    kwb = (df["rain"] - df["et0"]).resample("MS").sum()
    s3 = kwb.rolling(3).sum()
    out = pd.Series(np.nan, index=s3.index)
    for m in range(1, 13):
        v = s3[s3.index.month == m].dropna()
        if len(v) < 10:
            continue
        r = v.rank()
        p = (r - 0.44) / (len(v) + 0.12)
        out[v.index] = stats.norm.ppf(p)
    return out


def heuwetterfenster(d: pd.DataFrame, a) -> int:
    from .model import dry_window
    w = {"rain": d["rain"].to_numpy(), "et0": d["et0"].to_numpy()}
    doy = d.index.dayofyear.to_numpy()
    i, n = 0, 0
    while i < len(d):
        if 5 <= d.index[i].month <= 9:
            ok, _ = dry_window(w, i, a)
            if ok:
                n += 1
                i += a.trockentage_noetig
                continue
        i += 1
    return n


def kennzahlen(df: pd.DataFrame, jahre: list[int], a) -> pd.DataFrame:
    sp = spei3(df)
    rows = []
    for y in jahre:
        d = df[df.index.year == y]
        if len(d) < 360:
            continue
        r = {"jahr": y}
        r["vegetationsbeginn_gts200"] = gts_tag(d)
        warm = (d["tmean"] > 5).to_numpy()
        b = _runs(warm, 6)
        r["vegetationsperiode_beginn"] = float(d.index[b].dayofyear) if b is not None else np.nan
        h2 = d[d.index.month >= 7]
        e = _runs((h2["tmean"] < 5).to_numpy(), 6)
        r["vegetationsperiode_ende"] = float(h2.index[e].dayofyear) if e is not None else 365.0
        r["vegetationsperiode_laenge"] = r["vegetationsperiode_ende"] - r["vegetationsperiode_beginn"]
        r["sommertage"] = int((d["tmax"] >= 25).sum())
        r["hitzetage"] = int((d["tmax"] >= 30).sum())
        r["tropennaechte"] = int((d["tmin"] >= 20).sum())
        r["frosttage"] = int((d["tmin"] < 0).sum())
        vb = r["vegetationsbeginn_gts200"]
        if not np.isnan(vb):
            fr = d[(d.index.dayofyear > vb) & (d.index.month <= 6) & (d["tmin"] < 0)]
            r["letzter_frost_nach_vegbeginn"] = float(fr.index[-1].dayofyear) if len(fr) else np.nan
        so = d[(d.index.month >= 4) & (d.index.month <= 9)]
        r["laengste_trockenphase"], r["trockenphasen_14"] = _laengste_folge((so["rain"] < 1.0).to_numpy())
        r["kwb_apr_sep"] = float((so["rain"] - so["et0"]).sum())
        r["niederschlag_jahr"] = float(d["rain"].sum())
        r["tmittel_jahr"] = float(d["tmean"].mean())
        s = sp[(sp.index.year == y) & (sp.index.month >= 4) & (sp.index.month <= 9)]
        r["spei3_monate_unter_-1"] = int((s < -1).sum())
        r["heuwetterfenster"] = heuwetterfenster(d, a)
        rows.append(r)
    return pd.DataFrame(rows).set_index("jahr")


def modell_kennzahlen(sims: dict, traces: dict, clim: dict) -> pd.DataFrame:
    """Aus der ModVege-Simulation: Schnittzahl, Termin 1. Schnitt, Ertrag, Wasserstresstage (W unter 50 % des Bedarfs)."""
    from .model import _f_water
    rows = []
    for y, s in sims.items():
        tr = traces.get(y)
        stress = np.nan
        if tr is not None:
            W, pet = tr[:, 4], clim[y]["et0"][: len(tr)]
            stress = int(sum(_f_water(w, e) < 0.5 for w, e in zip(W, pet)))
        rows.append({"jahr": y, "schnitte_modell": len(s.cuts),
                     "erster_schnitt_tag": float(s.cuts[0].day + 1) if s.cuts else np.nan,
                     "ertrag_netto_dt": sum(c.netto_kg for c in s.cuts) / 100,
                     "wasserstresstage": stress})
    return pd.DataFrame(rows).set_index("jahr")


def trend(serie: pd.Series) -> dict:
    s = serie.dropna()
    if len(s) < 10 or s.nunique() < 3:
        return {"n": len(s)}
    x = s.index.to_numpy(float)
    tau, p = stats.kendalltau(x, s.to_numpy(float))
    sl, ic, lo, hi = stats.theilslopes(s.to_numpy(float), x, alpha=0.90)
    return {"n": len(s), "tau": tau, "p": p, "steigung_jahrzehnt": sl * 10, "lo": lo * 10, "hi": hi * 10,
            "intercept": ic, "slope": sl, "gesichert": bool(p < 0.05) if not np.isnan(p) else False}


BESCHREIBUNG = {
    "vegetationsbeginn_gts200": ("Vegetationsbeginn (GTS 200)", "Tag im Jahr", "früher = negativ"),
    "vegetationsperiode_laenge": ("Länge der thermischen Vegetationsperiode", "Tage", ""),
    "letzter_frost_nach_vegbeginn": ("Letzter Frost nach Vegetationsbeginn", "Tag im Jahr", ""),
    "sommertage": ("Sommertage (Tmax ≥ 25 °C)", "Tage", ""),
    "hitzetage": ("Hitzetage (Tmax ≥ 30 °C)", "Tage", ""),
    "tropennaechte": ("Tropennächte (Tmin ≥ 20 °C)", "Tage", ""),
    "frosttage": ("Frosttage (Tmin < 0 °C)", "Tage", ""),
    "laengste_trockenphase": ("Längste Trockenphase Apr–Sep (< 1 mm)", "Tage", ""),
    "trockenphasen_14": ("Trockenphasen ≥ 14 Tage (Apr–Sep)", "Anzahl", ""),
    "kwb_apr_sep": ("Klimatische Wasserbilanz Apr–Sep", "mm", ""),
    "spei3_monate_unter_-1": ("Monate mit SPEI-3 < −1 (Apr–Sep)", "Monate", ""),
    "heuwetterfenster": ("Heuwetterfenster Mai–Sep", "Anzahl", ""),
    "tmittel_jahr": ("Jahresmitteltemperatur", "°C", ""),
    "niederschlag_jahr": ("Jahresniederschlag", "mm", ""),
    "schnitte_modell": ("Mögliche Schnitte (ModVege)", "Anzahl", ""),
    "erster_schnitt_tag": ("Termin 1. Schnitt (ModVege)", "Tag im Jahr", ""),
    "ertrag_netto_dt": ("Heuertrag netto (ModVege)", "dt TM/ha", ""),
    "wasserstresstage": ("Wasserstresstage Gras (ModVege)", "Tage", ""),
    "hitzestunden_68": ("THI-Hitzestunden (≥ 68, Mai–Sep)", "Stunden", ""),
}
