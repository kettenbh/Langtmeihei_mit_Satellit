"""Modul 3: Futterqualität (Energie in MJ NEL, Rohprotein XP) gekoppelt an ModVege.

Energie:  OMD (ModVege) -> DOMD = OMD·(1000 − XA) [g/kg TM] -> ME = 0,016·DOMD (MAFF 1975)
          -> NEL = 0,6·[1 + 0,004·(q − 57)]·ME mit q = ME/GE·100, GE = 18,4 MJ/kg TM (GfE 2001)
Protein:  kritische N-Verdünnungskurve für C3-Gräser Nc = 4,8·W^−0,32 [% TM], W in t TM/ha (Lemaire & Salette 1984),
          N_Gras = NI·Nc (NI aus ModVege ≈ Stickstoff-Ernährungsindex), Mischung mit Klee und Totmasse, XP = 6,25·N.
Heu:      Abschlag für Bröckel- und Auswaschungsverluste je nach Verfahren und Regentagen (ANNAHMEN, siehe HEU_ABSCHLAG).
Kalibrierung: mittlere Abweichung zu eigenen Futteranalysen (data/futteranalysen.csv).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GE = 18.4
# Gras -> Heu: Abschläge [MJ NEL/kg TM bzw. g XP/kg TM]. ANNAHMEN in der Größenordnung der Differenz
# Grünfutter/Heu in Futterwerttabellen; mit Betriebsanalysen kalibrieren.
HEU_ABSCHLAG = {
    "boden": {"nel": 0.40, "nel_regentag": 0.15, "xp": 5.0, "xp_regentag": 5.0},
    "belueftung": {"nel": 0.20, "nel_regentag": 0.10, "xp": 3.0, "xp_regentag": 4.0},
}


def nel_aus_omd(omd, xa: float = 95.0):
    omd = np.asarray(omd, float)
    domd = omd * (1000.0 - xa)
    me = 0.016 * domd
    q = me / GE * 100
    return 0.6 * (1 + 0.004 * (q - 57)) * me


def me_aus_omd(omd, xa: float = 95.0):
    return 0.016 * np.asarray(omd, float) * (1000.0 - xa)


def n_kritisch(w_t_ha):
    w = np.maximum(np.asarray(w_t_ha, float), 1e-6)
    return np.where(w < 1.0, 4.8, 4.8 * w ** -0.32)


def xp_mischung(gruen_kg, tot_kg, ni: float, klee: float, n_leg: float, n_tot: float,
                w_gruen_bestand_kg=None):
    """Rohprotein [g/kg TM] der geernteten Masse.

    gruen_kg/tot_kg: geerntete grüne bzw. tote Masse [kg TM/ha]; die Verdünnung richtet sich nach der
    grünen Bestandesmasse (w_gruen_bestand_kg, Standard = gruen_kg)."""
    gruen_kg = np.asarray(gruen_kg, float)
    tot_kg = np.asarray(tot_kg, float)
    wb = gruen_kg if w_gruen_bestand_kg is None else np.asarray(w_gruen_bestand_kg, float)
    n_gras = np.minimum(ni, 1.2) * n_kritisch(wb / 1000.0)
    n_gruen = (1 - klee) * n_gras + klee * n_leg
    tot = gruen_kg + tot_kg
    n = np.where(tot > 0, (gruen_kg * n_gruen + tot_kg * n_tot) / np.maximum(tot, 1e-9), np.nan)
    return 6.25 * n * 10.0


def heu_abschlag(nel, xp, regentage, belueftung: bool):
    p = HEU_ABSCHLAG["belueftung" if belueftung else "boden"]
    return (np.asarray(nel) - p["nel"] - p["nel_regentag"] * np.asarray(regentage),
            np.asarray(xp) - p["xp"] - p["xp_regentag"] * np.asarray(regentage))


def stadium(st: float, st1: float, st2: float, schnitt_nr: int, tage_seit_schnitt: int | None) -> str:
    """Grobes Entwicklungsstadium: 1. Aufwuchs über Temperatursumme, Folgeaufwüchse über Aufwuchsdauer."""
    if schnitt_nr <= 1:
        if st < st1 - 150:
            return "Schossen"
        if st < st1 + 100:
            return "Ähren-/Rispenschieben"
        if st < st1 + 300:
            return "Beginn Blüte"
        if st < st2:
            return "Mitte bis Ende Blüte"
        return "nach der Blüte (überständig)"
    t = tage_seit_schnitt or 0
    if t < 35:
        return "junger Aufwuchs (< 5 Wochen)"
    if t < 49:
        return "Folgeaufwuchs 5–7 Wochen"
    if t < 63:
        return "Folgeaufwuchs 7–9 Wochen"
    return "Folgeaufwuchs > 9 Wochen (überständig)"


def geerntet(s, p, h: float):
    """Geerntete Kompartimente über Schnitthöhe h (wie model.harvestable)."""
    r = (h * p.BDGV * 10, h * p.BDGR * 10, h * p.BDDV * 10, h * p.BDDR * 10)
    gv, gr = max(0.0, s.GV - r[0]), max(0.0, s.GR - r[1])
    dv, dr = max(0.0, s.DV - r[2]), max(0.0, s.DR - r[3])
    return gv + gr, dv + dr


def xp_bestand(s, p, a, ni_faktor: float = 1.0) -> float:
    gruen, tot = geerntet(s, p, a.schnitthoehe_m)
    return float(xp_mischung(gruen, tot, p.NI * ni_faktor, a.kleeanteil, a.n_leguminosen_pct, a.n_totmasse_pct,
                             w_gruen_bestand_kg=s.GV + s.GR))


# ---------------------------------------------------------------------------
# Zielkonflikt Menge/Qualität über kommende Schnitttage (alle Wetterszenarien)
# ---------------------------------------------------------------------------
def schnittkurve(cur, clim: dict, stichtag: int, p, a, horizon: int = 45, perfect_days: int = 0,
                 korrektur: tuple[float, float] = (0.0, 0.0), ni_faktor: float = 1.0) -> pd.DataFrame:
    """Je Kandidatentag: Quantile von Masse, NEL, XP (als Heu) und Energieertrag [GJ NEL/ha] über alle Szenarien.

    Es wird bis zum Kandidatentag nicht gemäht (Bestand wächst weiter)."""
    from .forecast import _scenario_weather
    from .model import harvestable, run, dry_window

    s0 = cur.snapshots[min(stichtag, cur.n - 1)]
    rows = {h: [] for h in range(0, horizon + 1)}
    for y in sorted(clim):
        w = _scenario_weather(cur, clim[y], stichtag, perfect_days)
        s = s0.copy()
        for h in range(0, horizon + 1):
            d = stichtag + h
            if d >= 365 - a.trockentage_noetig:
                break
            if h > 0:
                run(w, p, a, state=s, start=d, end=d + 1, manage=False)
            H, omd, _ = harvestable(s, p, a.schnitthoehe_m)
            if H <= 0:
                continue
            ok, rt = dry_window(w, d, a)
            nel = float(nel_aus_omd(omd, a.rohasche_g_kg)) + korrektur[0]
            xp = xp_bestand(s, p, a, ni_faktor) + korrektur[1]
            nel_h, xp_h = heu_abschlag(nel, xp, rt, a.belueftung)
            verlust = min(0.6, a.feldverlust_basis + a.feldverlust_je_regentag * rt)
            netto = H * (1 - verlust)
            rows[h].append((H / 100, netto / 100, float(nel_h), float(xp_h), netto * float(nel_h) / 1000.0, ok,
                            s.ST, s.n_cuts, d - s.last_cut))
    out = []
    for h, v in rows.items():
        if len(v) < 5:
            continue
        arr = np.array(v, dtype=float)
        r = {"tag": stichtag + h, "n": len(v), "p_heuwetter": float(arr[:, 5].mean())}
        for j, name in enumerate(["brutto_dt", "netto_dt", "nel", "xp", "energie_gj"]):
            for q in (10, 50, 90):
                r[f"{name}_q{q}"] = float(np.percentile(arr[:, j], q))
        r["st"] = float(np.median(arr[:, 6]))
        r["schnitt_nr"] = int(np.median(arr[:, 7])) + 1
        r["tage_seit"] = int(np.median(arr[:, 8]))
        out.append(r)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Kalibrierung an eigenen Futteranalysen
# ---------------------------------------------------------------------------
ANALYSEN = Path(__file__).resolve().parent.parent / "data" / "futteranalysen.csv"


def lade_analysen() -> pd.DataFrame | None:
    if not ANALYSEN.exists():
        return None
    df = pd.read_csv(ANALYSEN, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datum"] = pd.to_datetime(df["datum"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["datum"])


def korrektur_aus_analysen(analysen: pd.DataFrame, modell: pd.DataFrame) -> tuple[float, float, int]:
    """Mittlere Abweichung Analyse − Modell (NEL, XP) über Schnitte, die nach Datum (±10 Tage) zugeordnet werden.

    modell: DataFrame mit Spalten datum, nel_heu, xp_heu (simulierte Schnitte der Historie)."""
    if analysen is None or modell is None or modell.empty:
        return 0.0, 0.0, 0
    d_nel, d_xp = [], []
    for _, r in analysen.iterrows():
        dist = (modell["datum"] - r["datum"]).abs()
        i = dist.idxmin()
        if dist[i] <= pd.Timedelta(days=10):
            if "nel" in r and pd.notna(r["nel"]):
                d_nel.append(float(r["nel"]) - float(modell.loc[i, "nel_heu"]))
            if "xp" in r and pd.notna(r["xp"]):
                d_xp.append(float(r["xp"]) - float(modell.loc[i, "xp_heu"]))
    n = max(len(d_nel), len(d_xp))
    return (float(np.mean(d_nel)) if d_nel else 0.0), (float(np.mean(d_xp)) if d_xp else 0.0), n


# ---------------------------------------------------------------------------
# Hybrid: Zielkonflikt Menge/Qualität aus dem Analyse-Ensemble (Modell + Sentinel-2)
# ---------------------------------------------------------------------------
def _kurve_aus_ensemble(S, P, w_ist: dict, clim: dict, stichtag: int, a, horizon: int, korrektur,
                        min_masse_kg: float = 300.0, ni_faktor: float = 1.0) -> pd.DataFrame:
    """S, P: Ensemble-Zustand/Parameter am Ende des Stichtags (n Mitglieder). Jedes Mitglied läuft mit jedem
    Klimajahr als Restwetter (n × k Läufe auf einmal), ohne Schnitt bis zum Kandidatentag."""
    from . import ensemble as E
    jahre = sorted(clim)
    k, n = len(jahre), S.n
    scen = {key: np.stack([np.concatenate([w_ist[key][:stichtag + 1], clim[y][key][stichtag + 1:]]) for y in jahre])
            for key in w_ist}                                               # (k, 365)
    ok_s, rt_s = E.dry_windows(scen["rain"], scen["et0"], a)                # (k, 365)
    W = {key: np.tile(v, (n, 1)) for key, v in scen.items()}                # Zeile i -> Szenario i % k
    S = S.repeat(k)
    P = P.repeat(k)
    sz = np.tile(np.arange(k), n)
    out = []
    for h in range(0, horizon + 1):
        d = stichtag + h
        if d >= 365 - a.trockentage_noetig:
            break
        if h > 0:
            E.run_ensemble(W, P, a, state=S, start=d, end=d + 1, manage=False)
        H, omd, rest = E.harvestable(S, P, a.schnitthoehe_m)
        m = H > min_masse_kg
        if m.sum() < min(5, len(m)):
            continue
        gruen = np.maximum(0, S.GV - rest[0]) + np.maximum(0, S.GR - rest[1])
        tot = np.maximum(0, S.DV - rest[2]) + np.maximum(0, S.DR - rest[3])
        rt = rt_s[sz, d]
        nel = nel_aus_omd(omd, a.rohasche_g_kg) + korrektur[0]
        xp = xp_mischung(gruen, tot, P.NI * ni_faktor, a.kleeanteil, a.n_leguminosen_pct, a.n_totmasse_pct,
                         w_gruen_bestand_kg=S.GV + S.GR) + korrektur[1]
        nel_h, xp_h = heu_abschlag(nel, xp, rt, a.belueftung)
        verlust = np.minimum(0.6, a.feldverlust_basis + a.feldverlust_je_regentag * rt)
        netto = H * (1 - verlust)
        # Qualität nur über Mitglieder mit nennenswerter Masse (direkt nach dem Schnitt bestimmt sonst die
        # Stoppel die Werte); Anteil erntewürdiger Mitglieder als eigene Spalte
        r = {"tag": d, "n": int(m.sum()), "p_heuwetter": float(ok_s[:, d].mean()),
             "p_erntewuerdig": float(np.mean(H >= a.min_ertrag_kg_tm_ha))}
        for name, v, alle in (("brutto_dt", H / 100, True), ("netto_dt", netto / 100, True), ("nel", nel_h, False),
                              ("xp", xp_h, False), ("energie_gj", netto * nel_h / 1000.0, True)):
            for q in (10, 50, 90):
                r[f"{name}_q{q}"] = float(np.percentile(v if alle else v[m], q))
        r["st"] = float(np.median(S.ST))
        r["schnitt_nr"] = int(np.median(S.n_cuts)) + 1
        r["tage_seit"] = int(np.median(d - S.last_cut))
        out.append(r)
    return pd.DataFrame(out)


def regel_schnitt_hypothesen(w: dict, p, a, stichtag: int, fenster: list, abstand: int = 20,
                             breite: int = 7) -> list[tuple[int, int]]:
    """Schnitttermine der Schnittregel (Ist-Wetter, App-Parameter) bis zum Stichtag, die kein erkanntes
    Schnittfenster in ±abstand Tagen haben, als Fenster ±breite. In einer Wolkenlücke sieht der Satellit
    einen Schnitt oft nicht; ein Betrieb lässt den ersten Aufwuchs aber kaum bis in den Juli stehen."""
    from . import ensemble as E
    P = E.EnsembleParams.from_params(p, 1)
    S = E.run_ensemble({k: v[None, :stichtag + 1] for k, v in w.items()}, P, a, end=stichtag + 1, manage=True)
    out = []
    for c in S.cut_day[0]:
        c = int(c)
        if c < 0 or c > stichtag:
            continue
        if any(f - abstand <= c <= sp + abstand for f, sp in fenster):
            continue
        out.append((c - breite, min(c + breite, stichtag)))
    return out


def schnittkurve_hybrid(daten, stichtag: int, clim: dict, p, a, horizon: int = 45, korrektur=(0.0, 0.0),
                        cfg=None, posterior=None, n: int = 32, seed: int = 0,
                        alter_korrigieren: bool = True, schnitt_resampling: int = 0,
                        schnitt_nachtragen: bool = True, regel_hypothesen: bool = True, ni_faktor: float = 1.0,
                        n_status=None, jahr: int | None = None):
    """Wie schnittkurve, aber der Bestand am Stichtag kommt aus der EnKF-Analyse mit Sentinel-2-LAI und den
    aus dem Satelliten erkannten Schnitten (statt aus der Schnittregel mit Ist-Wetter).

    daten: relativ.JahrDaten des laufenden Jahres (Wetter, Beobachtungen, Schnittfenster).
    Rückgabe: (Kurve wie schnittkurve, Info-Dict mit Aufnahmen, Schnitten, Analyse)."""
    from . import assimilation as DA_
    from . import beobachtung as B
    from .config import DA
    cfg = cfg or DA
    stichtag = int(min(stichtag, daten.n_ist - 1))
    o = daten.obs
    msk = o.tag <= stichtag
    obs = B.Beobachtungen(o.tag[msk], o.wert[msk], o.sigma[msk], o.clear[msk])
    fen = [(f, min(s_, stichtag)) for f, s_ in daten.fenster if f <= stichtag]
    hyp = regel_schnitt_hypothesen(daten.w, p, a, stichtag, fen) if regel_hypothesen else []
    an = DA_.analyse_saison(daten.w, p, a, obs, None, ende=stichtag + 1, cfg=cfg, seed=seed, posterior=posterior,
                            n=n if posterior is None else posterior.n, schnitt_fenster=fen,
                            alter_korrigieren=alter_korrigieren, schnitt_resampling=schnitt_resampling,
                            schnitt_nachtragen=schnitt_nachtragen, schnitt_hypothesen=hyp)
    lc = an.zustand.last_cut[an.zustand.last_cut >= 0]
    letzter = int(np.median(lc)) if len(lc) and np.mean(an.zustand.last_cut >= 0) >= 0.5 else None
    nf = None
    if n_status is not None and jahr is not None:
        from . import ndre as ND
        t = pd.Timestamp(jahr, 1, 1)
        nf = ND.n_faktor(n_status, t + pd.Timedelta(days=stichtag),
                         None if letzter is None else t + pd.Timedelta(days=letzter))
        ni_faktor = ni_faktor * nf.faktor
    kurve = _kurve_aus_ensemble(an.zustand, an.params, daten.w, clim, stichtag, a, horizon, korrektur,
                                ni_faktor=ni_faktor)
    info = {"aufnahmen": len(obs), "schnitte": fen, "hypothesen": hyp,
            "nachgetragen": an.nachgetragene_schnitte, "letzte_aufnahme": int(obs.tag.max()) if len(obs) else None,
            "analyse": an, "lai_heute": float(np.mean(an.zustand.lai(an.params))), "letzter_schnitt": letzter,
            "n_faktor": nf, "ni_faktor": ni_faktor}
    return kurve, info


def kurve_mischen(hybrid: pd.DataFrame, modell: pd.DataFrame, gewicht_hybrid: float = 0.5,
                  spalten=("nel", "xp")) -> pd.DataFrame:
    """Menge und Energieertrag aus dem Hybrid, Qualität (NEL, XP) als gewichtetes Mittel von Hybrid und Modell.
    Hintergrund: der Satellit misst die Menge (LAI), das Blattalter nur indirekt über erkannte Schnitte."""
    if hybrid.empty or modell.empty:
        return hybrid
    out = hybrid.copy()
    m = modell.set_index("tag").reindex(out["tag"])
    for name in spalten:
        for q in (10, 50, 90):
            c = f"{name}_q{q}"
            mv = m[c].to_numpy()
            out[c] = np.where(np.isfinite(mv), gewicht_hybrid * out[c].to_numpy() + (1 - gewicht_hybrid) * mv, out[c])
    # Energieertrag mit gemischter NEL neu (netto bleibt aus dem Hybrid)
    faktor = out["nel_q50"] / hybrid["nel_q50"].replace(0, np.nan)
    for q in (10, 50, 90):
        out[f"energie_gj_q{q}"] = hybrid[f"energie_gj_q{q}"] * faktor.fillna(1.0)
    return out


def schnittkurve_modell_ensemble(w_ist: dict, stichtag: int, clim: dict, p, a, horizon: int = 45,
                                 korrektur=(0.0, 0.0), ni_faktor: float = 1.0) -> pd.DataFrame:
    """Gegenstück ohne Satellit mit demselben Rechenweg: ein Mitglied, Schnittregel mit Ist-Wetter bis Stichtag."""
    from . import ensemble as E
    P = E.EnsembleParams.from_params(p, 1)
    S = E.run_ensemble({k: v[None, :] for k, v in w_ist.items()}, P, a, end=stichtag + 1, manage=True)
    return _kurve_aus_ensemble(S, P, w_ist, clim, stichtag, a, horizon, korrektur, ni_faktor=ni_faktor)
