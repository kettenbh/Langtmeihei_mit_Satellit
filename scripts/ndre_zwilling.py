"""Zwillingsexperiment: Rohprotein mit und ohne NDRE-N-Status (ohne Internet).

    python scripts/ndre_zwilling.py [--schnell]

Wahrheit je Feld wie in scripts/qualitaet_zwilling.py, zusätzlich schwankt der N-Ernährungsindex NI:
je Jahr um exp(N(0; 0,05)) und je Aufwuchs (zwischen zwei Schnitten, z. B. Gülle ja/nein) um exp(N(0; 0,08)).
Die synthetische S2-Reihe bekommt einen NDRE, der diesem N-Status folgt (heu.zwilling, Rauschen 0,10 wie in
Niederham gemessen). Die Referenzkurve CIre(LAI) wird je Feld aus den ANDEREN Jahren geschätzt.
Alle Verfahren kennen das Feldmittel von NI (wie nach der Feldkalibrierung), nicht die Schwankung.

ACHTUNG Zirkelschluss: Der NDRE ist hier nach genau der Annahme erzeugt, die heu.ndre umkehrt (Chlorophyll
proportional zum N-Status). Das Ergebnis zeigt, was der NDRE bei diesem Rauschpegel höchstens bringen kann,
nicht, dass die Annahme stimmt. Das zeigen erst Futteranalysen.
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HEU_SYNTHETIC", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from heu import beobachtung as B  # noqa: E402
from heu import ensemble as E  # noqa: E402
from heu import ndre as ND  # noqa: E402
from heu import qualitaet as Q  # noqa: E402
from heu import relativ as R  # noqa: E402
from heu import satellite as SAT  # noqa: E402
from heu import zwilling as Z  # noqa: E402
from heu.config import ANNAHMEN as A  # noqa: E402
from heu.model import ModVegeParams  # noqa: E402
from heu.weather import synthetic_weather, year_arrays  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relativ_zwilling import feldwetter  # noqa: E402

FELDER = 3 if "--schnell" in sys.argv else 6
JAHRE = list(range(2017, 2026))
STICHTAGE = {"10. Mai": 129, "25. Juni": 175, "10. Aug.": 221}
H = 14

df = synthetic_weather(1990, dt.date(2025, 12, 31), seed=11)
clim = {y: year_arrays(df, y) for y in range(1991, 2017)}
rows = []
for k in range(FELDER):
    rng = np.random.default_rng(500 + k)
    ni_feld = rng.uniform(0.5, 0.9)
    p_w = Z.wahre_parameter(ModVegeParams(NI=ni_feld, WHC=A.nutzbare_feldkapazitaet_mm), NI=ni_feld,
                            WHC=rng.uniform(70, 160), RUEmax=rng.uniform(2.6, 3.4))
    p_bekannt = ModVegeParams(NI=ni_feld, WHC=A.nutzbare_feldkapazitaet_mm)   # wie nach Feldkalibrierung
    feld = {}
    for y in JAHRE:
        w = year_arrays(df, y)
        wf = feldwetter(w, rng)
        Pw0 = E.EnsembleParams.from_params(p_w, 1)
        s_r = E.run_ensemble(wf, Pw0, A, manage=True)
        tage = sorted(int(t) + int(rng.integers(-10, 11)) for t in s_r.cut_day[0] if t >= 0)
        # N-Status je Aufwuchs: Grenzen = Schnitttage
        grenzen = [0] + tage + [365]
        f_jahr = np.exp(rng.normal(0, 0.05))
        ni_rel = np.ones(365)
        for g0, g1 in zip(grenzen[:-1], grenzen[1:]):
            ni_rel[g0:g1] = f_jahr * np.exp(rng.normal(0, 0.08))
        # Wahrheit abschnittsweise mit eigenem NI rechnen
        S = None
        lai = []
        zust = {}
        for g0, g1 in zip(grenzen[:-1], grenzen[1:]):
            P = E.EnsembleParams.from_params(p_w, 1)
            P.NI = P.NI * ni_rel[g0]
            for t in STICHTAGE.values():
                if g0 <= t < g1:
                    S_t, _ = E.run_ensemble(wf, P, A, state=None if S is None else S.copy(), start=g0, end=t + 1,
                                            manage=False, forced_cuts=np.array(tage)[None, :], trace=True)
                    zust[t] = (S_t, P)
            S, tr = E.run_ensemble(wf, P, A, state=S, start=g0, end=g1, manage=False,
                                   forced_cuts=np.array(tage)[None, :], trace=True)
            lai.append(tr["lai"][:, 0])
        lai = np.concatenate(lai)
        s2 = Z.synthetische_s2_reihe(lai, y, rng, ni_rel=ni_rel)
        feld[y] = (w, wf, s2, zust, ni_rel)
    s2_alle = pd.concat([v[2] for v in feld.values()])
    for y, (w, wf, s2, zust, ni_rel) in feld.items():
        if y < 2019:
            continue
        ref = ND.referenz(s2_alle, ohne_jahr=y)
        status = ND.n_status(s2, ref)
        cuts = SAT.detect_cuts(s2)
        tab = B.aus_s2(s2, list(cuts["datum"]) if len(cuts) else None)
        jd = R.JahrDaten(w, 365, B.jahr(tab, y), R.fenster_aus_schnitten(cuts, y))
        for name, t in STICHTAGE.items():
            Sw, Pw = zust[t]
            kw = Q._kurve_aus_ensemble(Sw, Pw, wf, {0: wf}, t, A, 20, (0.0, 0.0))
            if kw.empty:
                continue
            kw = kw.set_index("tag")
            km = Q.schnittkurve_modell_ensemble(w, t, clim, p_bekannt, A, horizon=20)
            if km.empty:
                continue
            km = km.set_index("tag")
            kh, _ = Q.schnittkurve_hybrid(jd, t, clim, p_bekannt, A, horizon=20, n=24, seed=k * 1000 + y)
            kn, info = Q.schnittkurve_hybrid(jd, t, clim, p_bekannt, A, horizon=20, n=24, seed=k * 1000 + y,
                                             n_status=status, jahr=y)
            km_n = Q.schnittkurve_modell_ensemble(w, t, clim, p_bekannt, A, horizon=20,
                                                  ni_faktor=info["ni_faktor"]).set_index("tag")
            if kh.empty or kn.empty:
                continue
            kh = Q.kurve_mischen(kh, km.reset_index()).set_index("tag")
            kn = Q.kurve_mischen(kn, km_n.reset_index()).set_index("tag")
            d = t + H
            if any(d not in x.index for x in (kw, km, kh, kn)):
                continue
            rows.append({"feld": k, "jahr": y, "stichtag": name, "xp_wahr": kw.loc[d, "xp_q50"],
                         "xp_modell": km.loc[d, "xp_q50"], "xp_hybrid": kh.loc[d, "xp_q50"],
                         "xp_ndre": kn.loc[d, "xp_q50"], "ni_rel": ni_rel[t], "faktor": info["ni_faktor"],
                         "n_aufn": info["n_faktor"].n})
d = pd.DataFrame(rows)
print(f"{FELDER} Felder × {len(JAHRE) - 2} Jahre, Rohprotein bei Schnitt in {H} Tagen [g/kg TM]\n")
print(f"{'Stichtag':9s} {'Modell':>7s} {'Hybrid':>7s} {'+NDRE':>7s}   Korr(Faktor, wahrer N-Status)  Aufn.")
for name in STICHTAGE:
    x = d[d["stichtag"] == name]
    e = {m: (x[f"xp_{m}"] - x["xp_wahr"]).abs().mean() for m in ("modell", "hybrid", "ndre")}
    r = np.corrcoef(np.log(x["faktor"]), np.log(x["ni_rel"]))[0, 1]
    print(f"{name:9s} {e['modell']:>7.1f} {e['hybrid']:>7.1f} {e['ndre']:>7.1f}   {r:>28.2f}  {x['n_aufn'].mean():5.1f}")
print(f"\nStreuung des wahren N-Status: {np.std(np.log(d['ni_rel'])):.3f} (log)")
