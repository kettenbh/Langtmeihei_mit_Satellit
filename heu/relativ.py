"""Relative Ertragsschätzung: Jahresertrag in % vom Mittel der übrigen Jahre (ohne Ballenzahlen).

Idee: Alles, was ein Feld jedes Jahr gleich verzerrt (Ertragspegel, LAI-Umrechnung, Bodengüte, systematische
Feldverluste), kürzt sich im Verhältnis "Jahr / Mittel der anderen Jahre" heraus. Das Modell muss nur die
Unterschiede zwischen den Jahren treffen.

Drei Schätzer je Jahr, jeweils relativ ("leave one out": das Jahr selbst zählt nicht zum Mittel):
  wetter   ModVege mit Wetterpunkt und Schnittregel, ohne Satellit (das bisherige Verfahren der App)
  hybrid   EnKF-Analyse: ModVege, S2-LAI assimiliert, Schnitte an den erkannten Terminen
  licht    reiner Satellitenindex: absorbierte Strahlung Σ PAR·(1 − exp(−k·LAI_S2)) von April bis Oktober,
           LAI linear zwischen den klaren Aufnahmen interpoliert (Monteith: Zuwachs ∝ absorbiertes Licht)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import assimilation as DA_
from . import beobachtung as B
from . import ensemble as E
from .config import DA, Annahmen, DAKonfig
from .model import ModVegeParams
from .weather import doy365

SAISON = (90, 305)          # 1. April bis 31. Oktober (Tag 0..364)


def relativ_loo(werte: dict[int, float]) -> dict[int, float]:
    """{Jahr: Wert} -> {Jahr: 100 · Wert / Mittel der übrigen Jahre}."""
    jahre = list(werte)
    out = {}
    for y in jahre:
        rest = [werte[j] for j in jahre if j != y]
        out[y] = 100.0 * werte[y] / np.mean(rest) if rest and np.mean(rest) > 0 else np.nan
    return out


def relativ_zu(werte: dict[int, float], basis: dict[int, float]) -> dict[int, float]:
    """Werte relativ zum Mittel einer Basis (z. B. 2026 gegen 2017–2025)."""
    m = float(np.mean(list(basis.values())))
    return {y: 100.0 * v / m for y, v in werte.items()}


# ---------------------------------------------------------------------------
# Schätzer
# ---------------------------------------------------------------------------
def ertrag_wetter(w: dict, p: ModVegeParams, a: Annahmen) -> float:
    """Brutto-Jahresertrag [kg TM/ha] mit Schnittregel, nur Wetter."""
    s = E.run_ensemble(w, E.EnsembleParams.from_params(p, 1), a, manage=True)
    return float(s.season_total(netto=False)[0])


def ertrag_hybrid(w: dict, p: ModVegeParams, a: Annahmen, obs: B.Beobachtungen, fenster: list,
                  cfg: DAKonfig = DA, posterior: E.EnsembleParams | None = None, n: int = 64,
                  seed: int = 0, groesse: str = "zuwachs") -> np.ndarray:
    """Jahreswert je Mitglied nach EnKF-Analyse über das ganze Jahr.

    groesse="zuwachs": Σ Bruttozuwachs April–Oktober [kg TM/ha]. Hängt nicht davon ab, ob jeder Schnitt
    erkannt wurde: Ein verpasster Schnitt fehlt sonst komplett in der Erntesumme (die Analyse schneidet nur
    an erkannten Terminen), obwohl der Satellit den Nachwuchs sieht. Relativ ist das der robustere Weg.
    groesse="ernte": Summe der Brutto-Schnittmengen an den erkannten Terminen."""
    an = DA_.analyse_saison(w, p, a, obs, None, cfg=cfg, seed=seed, posterior=posterior, n=n,
                            schnitt_fenster=fenster)
    if groesse == "ernte":
        return an.zustand.season_total(netto=False)
    t0, t1 = SAISON
    return an.trace["gro"][t0:t1].sum(axis=0)


def lichtindex(obs: B.Beobachtungen, w: dict, k: float = 0.6, lai_a: float = 1.0) -> float:
    """Σ PAR·fAPAR über die Saison [MJ/m²], fAPAR = 1 − exp(−k·LAI_S2/a). NaN bei zu wenigen Aufnahmen."""
    t0, t1 = SAISON
    m = (obs.tag >= t0 - 30) & (obs.tag <= t1 + 30)
    if m.sum() < 6:
        return np.nan
    tage = np.arange(t0, t1)
    lai = np.interp(tage, obs.tag[m], obs.wert[m]) / lai_a
    return float(np.sum(0.48 * w["rad"][tage] * (1.0 - np.exp(-k * lai))))


def fenster_aus_schnitten(cuts: pd.DataFrame, jahr: int) -> list[tuple[int, int]]:
    if cuts is None or not len(cuts):
        return []
    c = cuts[cuts["datum"].dt.year == jahr]
    return sorted((doy365(r.frueh.date()), doy365(r.spaet.date())) for r in c.itertuples())


# ---------------------------------------------------------------------------
# Hindcast für ein echtes Feld
# ---------------------------------------------------------------------------
@dataclass
class Hindcast:
    tabelle: pd.DataFrame        # je Jahr: absolute Werte und % vom Mittel der übrigen Jahre
    hybrid_ens: dict             # Jahr -> Aufwuchs je Mitglied (kg TM/ha)


def zuwachs_wetter(w: dict, p: ModVegeParams, a: Annahmen) -> tuple[float, float]:
    """(Brutto-Erntesumme, Σ Bruttozuwachs April–Oktober) mit Schnittregel, nur Wetter [kg TM/ha]."""
    s, tr = E.run_ensemble(w, E.EnsembleParams.from_params(p, 1), a, manage=True, trace=True)
    return float(s.season_total(netto=False)[0]), float(tr["gro"][SAISON[0]:SAISON[1], 0].sum())


def hindcast(s2: pd.DataFrame, s1: pd.DataFrame | None, wetter_df: pd.DataFrame, jahre, p: ModVegeParams,
             a: Annahmen, cfg: DAKonfig = DA, posterior: E.EnsembleParams | None = None, n: int = 64,
             seed: int = 0, protokoll=None) -> Hindcast:
    """Schätzer je Grünlandjahr, relativ zum Mittel der übrigen Jahre (Spalten *_pct).

    wetter        Brutto-Erntesumme mit Schnittregel (bisheriges Verfahren der App)
    wetter_zw     Aufwuchs (Σ Bruttozuwachs April–Oktober) nur aus Wetter
    hybrid        Aufwuchs nach EnKF-Analyse mit S2-LAI und erkannten Schnitten (mit 10–90-%-Band)
    licht         Satellitenindex absorbiertes Licht
    kombi         Mittel aus hybrid und licht (im Zwillingsexperiment am stabilsten)
    """
    from . import satellite as SAT
    from .weather import year_arrays
    cuts = SAT.detect_cuts(s2, s1)
    nutzung = SAT.nutzung_je_jahr(s2)
    tab = B.aus_s2(s2, list(cuts["datum"]) if len(cuts) else None)
    jahre = [y for y in jahre if nutzung.get(y) == SAT.NUTZUNG_GRUENLAND and year_arrays(wetter_df, y) is not None]
    werte = {k: {} for k in ("wetter", "wetter_zw", "hybrid", "licht")}
    ens, rows = {}, []
    for i, y in enumerate(jahre):
        w = year_arrays(wetter_df, y)
        obs = B.jahr(tab, y)
        fen = fenster_aus_schnitten(cuts, y)
        werte["wetter"][y], werte["wetter_zw"][y] = zuwachs_wetter(w, p, a)
        ens[y] = ertrag_hybrid(w, p, a, obs, fen, cfg, posterior, n, seed + y)
        werte["hybrid"][y] = float(np.mean(ens[y]))
        werte["licht"][y] = lichtindex(obs, w, lai_a=cfg.operator_a)
        rows.append({"jahr": y, "aufnahmen": len(obs), "schnitte_erkannt": len(fen),
                     "wetter_dt": werte["wetter"][y] / 100, "wetter_zw_dt": werte["wetter_zw"][y] / 100,
                     "hybrid_dt": werte["hybrid"][y] / 100, "licht_mj": werte["licht"][y]})
        if protokoll:
            protokoll(i + 1, len(jahre))
    df = pd.DataFrame(rows).set_index("jahr")
    for name, d in werte.items():
        r = relativ_loo({y: v for y, v in d.items() if np.isfinite(v)})
        df[f"{name}_pct"] = [r.get(y, np.nan) for y in df.index]
    df["kombi_pct"] = df[["hybrid_pct", "licht_pct"]].mean(axis=1)
    for y in df.index:
        rest = np.mean([werte["hybrid"][j] for j in df.index if j != y]) if len(df) > 1 else np.nan
        df.loc[y, "hybrid_p10"], df.loc[y, "hybrid_p90"] = np.percentile(ens[y], [10, 90]) / rest * 100
    return Hindcast(df, ens)


# ---------------------------------------------------------------------------
# Laufendes Jahr: relativer Aufwuchs zum Stichtag (Band verengt sich zur Ernte hin)
# ---------------------------------------------------------------------------
@dataclass
class JahrDaten:
    w: dict                      # Wetter-Arrays (365 Tage; im laufenden Jahr ab n_ist mit Klimamittel gefüllt)
    n_ist: int                   # Tage mit gemessenem Wetter
    obs: B.Beobachtungen
    fenster: list


def jahr_daten(wetter_df: pd.DataFrame, s2: pd.DataFrame, s1, jahre, clim: dict) -> dict[int, JahrDaten]:
    from . import satellite as SAT
    from .weather import partial_year
    cuts = SAT.detect_cuts(s2, s1)
    tab = B.aus_s2(s2, list(cuts["datum"]) if len(cuts) else None)
    mean = {k: np.mean([c[k] for c in clim.values()], axis=0) for k in next(iter(clim.values()))}
    out = {}
    for y in jahre:
        w, n_ist = partial_year(wetter_df, y)
        if n_ist < 60:
            continue
        w = {k: np.concatenate([w[k][:n_ist], mean[k][n_ist:]]) for k in mean}
        out[y] = JahrDaten(w, min(n_ist, 365), B.jahr(tab, y), fenster_aus_schnitten(cuts, y))
    return out


def feste_mitglieder(p: ModVegeParams, a: Annahmen, n: int, cfg: DAKonfig = DA, seed: int = 0,
                     posterior: E.EnsembleParams | None = None) -> E.EnsembleParams:
    """Ein Parameter-Ensemble, das in allen Jahren gleich verwendet wird. Dann kürzt sich die
    Parameterunsicherheit im Verhältnis Jahr / Mittel heraus, Mitglied für Mitglied."""
    if posterior is not None:
        return posterior.take(np.random.default_rng(seed).integers(0, posterior.n, n))
    return DA_.prior(p, a, np.random.default_rng(seed), cfg, n=n)[0]


def _aufwuchs_bis_ende(jd: JahrDaten, stichtag: int, rest: list[dict], p, a, cfg, ep, seed) -> np.ndarray:
    """Analyse bis Stichtag, danach je Restwetter Schnittregel. (n, len(rest)) Aufwuchs April–Oktober."""
    t0, t1 = SAISON
    o = jd.obs
    m = o.tag <= stichtag
    o = B.Beobachtungen(o.tag[m], o.wert[m], o.sigma[m], o.clear[m])
    fen = [(f, min(s_, stichtag)) for f, s_ in jd.fenster if f <= stichtag]
    an = DA_.analyse_saison(jd.w, p, a, o, None, ende=stichtag + 1, cfg=cfg, seed=seed, posterior=ep,
                            n=ep.n, schnitt_fenster=fen)
    bisher = an.trace["gro"][t0:min(t1, stichtag + 1)].sum(axis=0) if stichtag >= t0 else np.zeros(ep.n)
    if stichtag + 1 >= t1:
        return np.repeat(bisher[:, None], len(rest), axis=1), bisher, len(o), len(fen)
    cols = []
    for rw in rest:
        w = {k: np.concatenate([jd.w[k][:stichtag + 1], rw[k][stichtag + 1:]]) for k in jd.w}
        _, tr = E.run_ensemble(w, an.params.take(np.arange(ep.n)), a, state=an.zustand.copy(),
                               start=stichtag + 1, end=t1, manage=True, trace=True)
        cols.append(bisher + tr["gro"][max(0, t0 - stichtag - 1):].sum(axis=0))
    return np.stack(cols, axis=1), bisher, len(o), len(fen)


@dataclass
class RelativPrognose:
    stichtag: int
    prozent: np.ndarray          # Stichproben (Mitglieder × Wetterjahre) in % vom Mittel der Basisjahre
    bisher_prozent: float        # bis zum Stichtag gewachsener Aufwuchs in % des Basismittels
    n_obs: int                   # verwendete S2-Aufnahmen bis zum Stichtag
    n_schnitte: int


def prognose_relativ(jahr: int, daten: dict[int, JahrDaten], basis_jahre, clim: dict[int, dict], stichtag: int,
                     p: ModVegeParams, a: Annahmen, ep: E.EnsembleParams, cfg: DAKonfig = DA,
                     seed: int = 0) -> RelativPrognose:
    """Aufwuchs April–Oktober des Jahres `jahr` in % vom Mittel der Basisjahre, Stand Stichtag.

    Das laufende Jahr: Analyse mit S2 bis zum Stichtag, danach jedes Klimajahr als mögliches Restwetter.
    Die Basisjahre werden mit *demselben Rechenweg* zum selben Stichtag gerechnet (Analyse bis Stichtag,
    danach ihr tatsächliches Wetter mit Schnittregel). So vergleicht das Verhältnis Gleiches mit Gleichem,
    und bei frühem Stichtag liegt der Median in einem Normaljahr bei etwa 100 %. Das Verhältnis wird je
    Mitglied gebildet (gleiche Parameter in allen Jahren)."""
    stichtag = int(min(stichtag, daten[jahr].n_ist - 1))
    basis = [y for y in basis_jahre if y in daten and y != jahr]
    b = np.mean([_aufwuchs_bis_ende(daten[y], stichtag, [daten[y].w], p, a, cfg, ep, seed + y)[0][:, 0]
                 for y in basis], axis=0)                                   # (n,)
    tot, bisher, n_o, n_f = _aufwuchs_bis_ende(daten[jahr], stichtag, list(clim.values()), p, a, cfg, ep, seed + jahr)
    return RelativPrognose(stichtag, (100.0 * tot / b[:, None]).ravel(), 100.0 * float(np.mean(bisher / b)), n_o, n_f)


def facher_relativ(jahr, daten, basis_jahre, clim, stichtage, p, a, ep, cfg: DAKonfig = DA, protokoll=None):
    """Prognosen für mehrere Stichtage -> DataFrame (Stichtag, p10, p25, p50, p75, p90, bisher)."""
    rows = []
    for i, t in enumerate(stichtage):
        r = prognose_relativ(jahr, daten, basis_jahre, clim, t, p, a, ep, cfg)
        q = np.percentile(r.prozent, [10, 25, 50, 75, 90])
        rows.append({"stichtag": r.stichtag, "p10": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p90": q[4],
                     "bisher": r.bisher_prozent, "aufnahmen": r.n_obs, "schnitte": r.n_schnitte})
        if protokoll:
            protokoll(i + 1, len(stichtage))
    return pd.DataFrame(rows)
