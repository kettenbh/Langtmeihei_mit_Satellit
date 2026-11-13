"""Modul 2: Parasiten- und Fliegendruck aus Wetterdaten (relative Risikoindizes, keine Diagnose).

Leberegel (Fasciola hepatica, Zwischenwirt Zwergschlammschnecke Galba truncatula)
  - Ollerenshaw-Index (Ollerenshaw & Rowlands 1959): Mt = n·(R − P + 5) je Monat Mai–Oktober,
    R = Niederschlag, P = potenzielle Verdunstung (beide in Zoll), n = Regentage.
    Monatswerte gedeckelt, Mai und Oktober halb gewichtet (Detailregeln aus Sekundärliteratur übernommen,
    vor produktiver Nutzung mit der Originalarbeit abgleichen).
  - Wasserbilanz-Gradtage Wb-GDD (nach Malone et al. 1998): Gradtage > 10 °C nur an Tagen, an denen die
    klimatische Wasserbilanz der letzten 30 Tage positiv ist (hier tägliche statt monatliche Variante).
Magen-Darm-Strongyliden (Ostertagia ostertagi) – vereinfacht nach GLOWORM-FL (Rose et al. 2015):
    Entwicklung Ei->L3: δ(T) = −0,07258 + 0,00976·T (Schwelle 7,44 °C)
    Sterberaten: μ1 = exp(−4,38278 − 0,10640·T + 0,00540·T²) (Ei–L2),
                 μ4 = exp(−6,388 − 0,2681·T + 0,01633·T² − 0,00016·T³) (L3 im Boden), μ3 = μ5 = 10·μ4
    Auswanderung aus dem Kotfladen: 0,06/Tag an Tagen mit ausreichend Regen (Rose 2015: ≥ 2 mm;
    Wang et al. 2022 für Ostertagia/Cooperia: ≥ 5 mm, hier Standard), vertikal Boden->Gras:
    m2 = exp(−5,48240 + 0,45392·T − 0,01252·T²).
    Die Koeffizienten stammen aus einer Zusammenfassung der Tabelle 2 und sind vor produktiver Nutzung
    mit dem Original abzugleichen. Eiausscheidung konstant je Tier (keine Rückkopplung Infektion -> Ausscheidung).
Weidefliegen
  - Hornfliege (Haematobia irritans): Generationen über Gradtage (siehe config.ParasitenAnnahmen), Tagesaktivität.
  - Kopffliege (Hydrotaea irritans, Sommermastitis): eine Generation, Flugzeit Juni–September, Tagesaktivität.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, ParasitenAnnahmen

ZOLL = 25.4
WEIDEPLAN = DATA_DIR / "weideplan.csv"


# ---------------------------------------------------------------------------
# Hilfen
# ---------------------------------------------------------------------------
def _ohne_29feb(s: pd.Series) -> pd.Series:
    return s[~((s.index.month == 2) & (s.index.day == 29))]


def jahresmatrix(s: pd.Series) -> pd.DataFrame:
    """Tägliche Reihe -> Matrix Jahr × Tag (0..364)."""
    s = _ohne_29feb(s)
    d = pd.DataFrame({"v": s.values, "jahr": s.index.year})
    d["tag"] = d.groupby("jahr").cumcount()
    return d.pivot(index="jahr", columns="tag", values="v")


def perzentil(wert: float, historie) -> float:
    h = np.asarray(historie, float)
    h = h[~np.isnan(h)]
    if len(h) == 0 or np.isnan(wert):
        return np.nan
    return float((np.sum(h < wert) + 0.5 * np.sum(h == wert)) / len(h) * 100)


def ampel(p: float) -> str:
    if np.isnan(p):
        return "–"
    return "🟢 niedrig" if p < 50 else ("🟡 erhöht" if p < 80 else "🔴 hoch")


def _md(s: str, year: int) -> pd.Timestamp:
    m, d = s.split("-")
    return pd.Timestamp(year, int(m), int(d))


# ---------------------------------------------------------------------------
# Leberegel
# ---------------------------------------------------------------------------
def ollerenshaw(df: pd.DataFrame, pa: ParasitenAnnahmen) -> pd.DataFrame:
    """Monatswerte Mt (Mai–Okt) je Jahr und kumulierter Sommerindex."""
    d = df[(df.index.month >= 5) & (df.index.month <= 10)]
    g = d.groupby([d.index.year, d.index.month])
    m = pd.DataFrame({
        "R": g["rain"].sum() / ZOLL, "P": g["et0"].sum() / ZOLL,
        "n": g["rain"].apply(lambda s: int((s >= pa.oll_regentag_mm).sum())),
    })
    m.index.names = ["jahr", "monat"]
    m["mt"] = (m["n"] * (m["R"] - m["P"] + 5)).clip(lower=0, upper=pa.oll_monat_max)
    halb = m.index.get_level_values("monat").isin([5, 10])
    m.loc[halb, "mt"] = m.loc[halb, "mt"] / 2
    m["kumuliert"] = m.groupby(level="jahr")["mt"].cumsum()
    return m


def wb_gdd(df: pd.DataFrame, pa: ParasitenAnnahmen) -> pd.Series:
    kwb30 = (df["rain"] - df["et0"]).rolling(pa.wbgdd_fenster_tage, min_periods=10).sum()
    gdd = (df["tmean"] - pa.wbgdd_basis_c).clip(lower=0).where(kwb30 > 0, 0.0)
    return gdd.groupby(df.index.year).cumsum()


# ---------------------------------------------------------------------------
# Magen-Darm-Würmer (Ostertagia), Kohortenmodell
# ---------------------------------------------------------------------------
def _delta(t):
    return np.maximum(0.0, -0.07258 + 0.00976 * t)


def _mu1(t):
    return np.exp(-4.38278 - 0.10640 * t + 0.00540 * t**2)


def _mu4(t):
    return np.exp(-6.388 - 0.2681 * t + 0.01633 * t**2 - 0.00016 * t**3)


def _m2(t):
    return np.exp(-5.48240 + 0.45392 * t - 0.01252 * t**2)


def mds_modell(df: pd.DataFrame, beweidung: pd.Series, regen_mm: float = 5.0, m1: float = 0.06,
               l3_start: float = 20.0) -> pd.DataFrame:
    """Durchgehende Tagessimulation (Überwinterung der L3 ergibt sich aus den Sterberaten). beweidung: Tiere auf der Fläche je Tag (gleicher Index wie df).

    Rückgabe: E (Ei–L2 im Fladen), L3f (L3 im Fladen), L3s (Boden), L3h (Gras = Infektionsdruck), relative Einheiten."""
    T = np.clip(df["tmean"].to_numpy(float), -5, 35)
    rain = df["rain"].to_numpy(float)
    tiere = beweidung.reindex(df.index).fillna(0).to_numpy(float)
    n = len(df)
    out = np.zeros((n, 4))
    E = L3f = L3s = 0.0
    L3h = l3_start
    for i in range(n):
        t = T[i]
        de, mu1 = 2 * _delta(t), _mu1(t)  # zwei Teilstadien -> Rate verdoppelt (Rose 2015)
        mu4 = _mu4(t)
        mu3 = 10 * mu4
        mig = m1 if rain[i] >= regen_mm else 0.0
        m2 = min(1.0, _m2(t))
        inp = tiere[i]
        dE = inp - (mu1 + de / 2) * E
        dL3f = de / 2 * E - (mu3 + mig) * L3f
        dL3s = mig * L3f - (mu4 + m2) * L3s
        dL3h = m2 * L3s - mu3 * L3h
        E, L3f, L3s, L3h = max(E + dE, 0), max(L3f + dL3f, 0), max(L3s + dL3s, 0), max(L3h + dL3h, 0)
        out[i] = (E, L3f, L3s, L3h)
    return pd.DataFrame(out, index=df.index, columns=["E", "L3f", "L3s", "L3h"])


def lade_weideplan() -> pd.DataFrame | None:
    if not WEIDEPLAN.exists():
        return None
    w = pd.read_csv(WEIDEPLAN, comment="#")
    w.columns = [c.strip().lower() for c in w.columns]
    return w


def beweidung_serie(index: pd.DatetimeIndex, plan: pd.DataFrame | None, weide: str, pa: ParasitenAnnahmen,
                    gruppe_filter: str = "jung") -> pd.Series:
    """Tiere der Jungrinder-Gruppen auf einer Weide je Tag. Der Plan (TT.MM.) wird auf jedes Jahr übertragen."""
    s = pd.Series(0.0, index=index)
    if plan is None:
        for y in sorted(set(index.year)):
            s[(s.index >= _md(pa.weide_austrieb, y)) & (s.index <= _md(pa.weide_abtrieb, y))] = 15.0
        return s
    for _, r in plan.iterrows():
        if str(r.get("weide", "")).strip() != weide or gruppe_filter not in str(r.get("gruppe", "")).lower():
            continue
        v, b = str(r["von"]).strip(), str(r["bis"]).strip()
        for y in sorted(set(index.year)):
            von = pd.Timestamp(dt.datetime.strptime(f"{v}.{y}", "%d.%m.%Y"))
            bis = pd.Timestamp(dt.datetime.strptime(f"{b}.{y}", "%d.%m.%Y"))
            s[(s.index >= von) & (s.index <= bis)] += float(r.get("tiere", 1))
    return s


# ---------------------------------------------------------------------------
# Fliegen
# ---------------------------------------------------------------------------
def _aktivitaet(df: pd.DataFrame, t_min: float, t_opt: float) -> pd.Series:
    ft = ((df["tmax"] - t_min) / (t_opt - t_min)).clip(0, 1)
    fr = np.where(df["rain"] >= 5, 0.5, 1.0)
    fw = (1 - (df["wind"].fillna(15) - 25) / 30).clip(0.3, 1.0)   # Tagesmaximum Wind km/h
    return ft * fr * fw


def hornfliege(df: pd.DataFrame, pa: ParasitenAnnahmen) -> pd.DataFrame:
    gt = (df["tmean"] - pa.hornfliege_basis_c).clip(lower=0)
    gt = gt.where(df.index.month >= 4, 0.0)
    gtk = gt.groupby(df.index.year).cumsum()
    gen = gtk / pa.hornfliege_gt_generation
    akt = _aktivitaet(df, 15.0, 25.0)
    druck = akt * np.minimum(gen, 4) / 4
    return pd.DataFrame({"gradtage": gtk, "generationen": gen, "aktivitaet": akt,
                         "druck": druck, "druck_kum": druck.groupby(df.index.year).cumsum()})


def kopffliege(df: pd.DataFrame, pa: ParasitenAnnahmen) -> pd.DataFrame:
    inside = pd.Series(False, index=df.index)
    for y in sorted(set(df.index.year)):
        inside[(df.index >= _md(pa.kopffliege_flug_von, y)) & (df.index <= _md(pa.kopffliege_flug_bis, y))] = True
    akt = (_aktivitaet(df, 14.0, 22.0) * inside).astype(float)
    return pd.DataFrame({"aktivitaet": akt, "druck_kum": akt.groupby(df.index.year).cumsum()})


# ---------------------------------------------------------------------------
# Gesamtübersicht: Indexreihen, Perzentile
# ---------------------------------------------------------------------------
def indexreihen(df: pd.DataFrame, pa: ParasitenAnnahmen, plan: pd.DataFrame | None, weiden: list[str]) -> dict[str, pd.Series]:
    """Kumulierte, tägliche Indexreihen (für Perzentil-Vergleich am gleichen Kalendertag)."""
    out = {}
    o = ollerenshaw(df, pa)
    # Ollerenshaw als Tagesreihe: kumulierter Wert bis zum Ende des Vormonats + laufender Monat anteilig
    daily = pd.Series(0.0, index=df.index)
    for (y, m), r in o.iterrows():
        ende = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
        daily[(daily.index >= ende) & (daily.index.year == y)] = r["kumuliert"]
    out["Leberegel (Ollerenshaw)"] = daily
    out["Leberegel (Wb-GDD)"] = wb_gdd(df, pa)
    for w in weiden:
        mds = mds_modell(df, beweidung_serie(df.index, plan, w, pa))
        out[f"Magen-Darm-Würmer {w}"] = mds["L3h"]
    out["Hornfliege"] = hornfliege(df, pa)["druck_kum"]
    out["Kopffliege"] = kopffliege(df, pa)["druck_kum"]
    return out


def risikotabelle(reihen: dict[str, pd.Series], stichtag: pd.Timestamp, hist_jahre: list[int]) -> pd.DataFrame:
    rows = []
    for name, s in reihen.items():
        m = jahresmatrix(s)
        tag = min(364, (dt.date(2001, stichtag.month, min(stichtag.day, 28 if stichtag.month == 2 else stichtag.day))
                        - dt.date(2001, 1, 1)).days)
        if stichtag.year not in m.index:
            continue
        wert = float(m.loc[stichtag.year, tag])
        hist = m.loc[[y for y in hist_jahre if y in m.index], tag]
        p = perzentil(wert, hist)
        rows.append({"Erreger": name, "Wert": wert, "Median 1991–2025": float(hist.median()),
                     "Perzentil": p, "Einstufung": ampel(p)})
    return pd.DataFrame(rows)


def terminhinweise(df: pd.DataFrame, pa: ParasitenAnnahmen, jahr: int, l3: pd.Series | None) -> list[str]:
    tips = []
    aus = _md(pa.weide_austrieb, jahr)
    tips.append(f"Kotprobe Jungrinder (MDS) ca. 6–8 Wochen nach Austrieb: {aus + pd.Timedelta(weeks=6):%d.%m.}–"
                f"{aus + pd.Timedelta(weeks=8):%d.%m.}{jahr}.")
    if l3 is not None:
        y = l3[(l3.index.year == jahr) & (l3.index >= aus)]
        if len(y) > 10:
            basis, spitze = float(y.iloc[0]), float(y.max())
            if spitze > basis * 1.2:
                ab = y[y >= basis + 0.5 * (spitze - basis)]
                tips.append(f"Larvendruck auf dem Gras (Modell) hat die Hälfte seines Anstiegs nach dem Austrieb ab "
                            f"{ab.index[0]:%d.%m.} erreicht: Umtrieb auf eine gemähte Fläche vorher planen, "
                            f"Kotprobe davor.")
    hf = hornfliege(df[df.index.year == jahr], pa)
    g1 = hf[hf["generationen"] >= 1]
    if len(g1):
        tips.append(f"Hornfliege: erste Generation laut Gradtagen ab ca. {g1.index[0]:%d.%m.}.")
    tips.append(f"Kopffliege (Sommermastitis): Schutz für Trockensteher und Kalbinnen vor "
                f"{_md(pa.kopffliege_flug_von, jahr):%d.%m.} anlegen.")
    return tips
