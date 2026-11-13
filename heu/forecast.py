"""Training (Kalibrierung, Klimatologie) und probabilistische Prognose für Stichtage."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import Annahmen
from .model import ModVegeParams, State, harvestable, is_ready, dry_window, run, season_total
from .weather import year_arrays, partial_year


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def training_years(df: pd.DataFrame, a: Annahmen) -> dict[int, dict]:
    out = {}
    for y in range(a.trainings_start, a.trainings_ende + 1):
        w = year_arrays(df, y)
        if w is not None:
            out[y] = w
    return out


def simulate_years(years: dict[int, dict], p: ModVegeParams, a: Annahmen) -> dict[int, State]:
    return {y: run(w, p, a) for y, w in years.items()}


def calibrate_ni(years: dict[int, dict], base: ModVegeParams, a: Annahmen) -> tuple[ModVegeParams, float]:
    """Bisektion über den Nährstoffindex NI, bis der mittlere Brutto-Jahresertrag
    der Trainingsjahre dem Kalibrierziel entspricht."""
    target = a.ziel_jahresertrag_dt_tm_ha * 100.0

    def mean_yield(ni: float) -> float:
        p = replace(base, NI=ni)
        return float(np.mean([season_total(s, netto=False) for s in simulate_years(years, p, a).values()]))

    lo, hi = 0.3, 1.2
    ylo, yhi = mean_yield(lo), mean_yield(hi)
    if target <= ylo:
        return replace(base, NI=lo), ylo / 100
    if target >= yhi:
        return replace(base, NI=hi), yhi / 100
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if mean_yield(mid) < target:
            lo = mid
        else:
            hi = mid
    ni = 0.5 * (lo + hi)
    return replace(base, NI=ni), mean_yield(ni) / 100


def history_table(sims: dict[int, State]) -> pd.DataFrame:
    rows = []
    for y, s in sims.items():
        r = {"jahr": y, "saison": season_total(s) / 100, "n_schnitte": len(s.cuts)}
        for i, c in enumerate(s.cuts[:5], start=1):
            r[f"schnitt{i}"] = c.netto_kg / 100
            r[f"tag{i}"] = c.day
            r[f"nel{i}"] = c.nel_heu
            r[f"xp{i}"] = c.xp_heu
        rows.append(r)
    return pd.DataFrame(rows).set_index("jahr")


# ---------------------------------------------------------------------------
# Laufendes Jahr: "Ist"-Simulation mit Zustands-Schnappschüssen
# ---------------------------------------------------------------------------
@dataclass
class CurrentYear:
    w: dict                 # Arrays bis letzter Datentag
    n: int                  # Anzahl verfügbarer Tage
    snapshots: list         # Zustand am ENDE jedes Tages d (Index d)
    final: State            # Ist-Simulation (Lücke nach letztem Datentag mit Klimamittel gefüllt)


def run_current_year(df: pd.DataFrame, year: int, clim: dict[int, dict], p: ModVegeParams, a: Annahmen) -> CurrentYear:
    w, n = partial_year(df, year)
    # Restjahr (nach letztem Datentag) mit dem Tagesmittel der Klimatologie füllen
    full = {}
    for k in w:
        mean = np.mean([c[k] for c in clim.values()], axis=0)
        full[k] = np.concatenate([w[k], mean[n:]])
    s = State(WR=p.WHC)
    snaps = []
    for d in range(n):
        run(full, p, a, state=s, start=d, end=d + 1)
        snaps.append(s.copy())
    run(full, p, a, state=s, start=n, end=365)
    return CurrentYear(w=w, n=n, snapshots=snaps, final=s)


def _scenario_weather(cur: CurrentYear, clim_w: dict, stichtag: int, perfect_days: int) -> dict:
    """Wetter = Ist bis Stichtag (+ optional 'perfekte Vorhersage'), danach Klimajahr."""
    cut = min(cur.n, stichtag + 1 + perfect_days)
    return {k: np.concatenate([cur.w[k][:cut], clim_w[k][cut:]]) for k in cur.w}


# ---------------------------------------------------------------------------
# Prognose zum Stichtag
# ---------------------------------------------------------------------------
@dataclass
class Prognose:
    stichtag: int
    saison: np.ndarray          # Stichproben Saisonertrag netto [dt TM/ha]
    schnitte: dict              # i -> Stichproben Schnitt i [dt TM/ha] (nan = kein Schnitt)
    schnitt_tage: dict          # i -> Szenario-Schnitttage (ohne Ziehungen)
    erledigt: list              # bereits erfolgte Schnitte (Cut-Objekte)
    szenario_jahre: list


def forecast(cur: CurrentYear, clim: dict[int, dict], stichtag: int, p: ModVegeParams, a: Annahmen,
             perfect_days: int = 0, seed: int = 0) -> Prognose:
    rng = np.random.default_rng(seed + stichtag)
    s0 = cur.snapshots[min(stichtag, cur.n - 1)]
    done = list(s0.cuts)
    done_total = sum(c.netto_kg for c in done) / 100
    k = a.ziehungen_je_szenario
    saison, per_cut, cut_days = [], {i: [] for i in range(1, a.max_schnitte + 1)}, {i: [] for i in range(1, a.max_schnitte + 1)}
    years = sorted(clim)
    for y in years:
        w = _scenario_weather(cur, clim[y], stichtag, perfect_days)
        s = run(w, p, a, state=s0.copy(), start=stichtag + 1, end=365)
        future = s.cuts[len(done):]
        # Strukturfehler je künftigem Schnitt (log-normal, mittelwerttreu)
        sig = a.modellfehler_sigma
        f = rng.lognormal(-0.5 * sig**2, sig, size=(k, max(1, len(future))))
        fut_vals = np.array([c.netto_kg / 100 for c in future]) if future else np.zeros(0)
        draws_future = (f[:, : len(future)] * fut_vals).sum(axis=1) if future else np.zeros(k)
        saison.append(done_total + draws_future)
        for i in range(1, a.max_schnitte + 1):
            if i <= len(done):
                per_cut[i].append(np.full(k, done[i - 1].netto_kg / 100))
                cut_days[i].append(done[i - 1].day)
            elif i - len(done) <= len(future):
                j = i - len(done) - 1
                per_cut[i].append(f[:, j] * fut_vals[j])
                cut_days[i].append(future[j].day)
            else:
                per_cut[i].append(np.full(k, np.nan))
                cut_days[i].append(np.nan)
    return Prognose(
        stichtag=stichtag,
        saison=np.concatenate(saison),
        schnitte={i: np.concatenate(v) for i, v in per_cut.items()},
        schnitt_tage={i: np.array(v, dtype=float) for i, v in cut_days.items()},
        erledigt=done,
        szenario_jahre=years,
    )


def fan(cur: CurrentYear, clim: dict, p: ModVegeParams, a: Annahmen, start: int, end: int, step: int,
        perfect_days: int = 0) -> pd.DataFrame:
    """Quantile der Prognose über viele Stichtage (für das Fächerdiagramm)."""
    rows = []
    for d in range(start, min(end, cur.n - 1) + 1, step):
        pr = forecast(cur, clim, d, p, a, perfect_days)
        r = {"tag": d}
        for name, arr in [("saison", pr.saison)] + [(f"schnitt{i}", v) for i, v in pr.schnitte.items()]:
            v = arr[~np.isnan(arr)]
            share = len(v) / len(arr)
            for q in (10, 25, 50, 75, 90):
                r[f"{name}_q{q}"] = np.percentile(v, q) if share >= 0.5 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Schnittzeitpunkt: Bewertung der kommenden Tage
# ---------------------------------------------------------------------------
def cut_timing(cur: CurrentYear, clim: dict, stichtag: int, p: ModVegeParams, a: Annahmen,
               horizon: int = 14, perfect_days: int = 0) -> pd.DataFrame:
    """Für jeden der nächsten Tage: Aufwuchs, Qualität, Heuwetter-Wahrscheinlichkeit
    über alle Wetterszenarien, ohne dass zwischendurch gemäht wird."""
    s0 = cur.snapshots[min(stichtag, cur.n - 1)]
    res = {h: {"H": [], "omd": [], "dry": [], "ready": []} for h in range(1, horizon + 1)}
    for y in sorted(clim):
        w = _scenario_weather(cur, clim[y], stichtag, perfect_days)
        s = s0.copy()
        for h in range(1, horizon + 1):
            d = stichtag + h
            if d >= 365 - a.trockentage_noetig:
                break
            H, omd, _ = harvestable(s, p, a.schnitthoehe_m)
            ok, _ = dry_window(w, d, a)
            res[h]["H"].append(H)
            res[h]["omd"].append(omd)
            res[h]["dry"].append(ok)
            res[h]["ready"].append(is_ready(s, d, H, a))
            run(w, p, a, state=s, start=d, end=d + 1, manage=False)
    rows = []
    for h, r in res.items():
        if not r["H"]:
            continue
        rows.append({
            "tag": stichtag + h,
            "ertrag_dt": np.median(r["H"]) / 100,
            "omd": np.median(r["omd"]),
            "p_heuwetter": float(np.mean(r["dry"])),
            "p_reif": float(np.mean(r["ready"])),
        })
    return pd.DataFrame(rows)
