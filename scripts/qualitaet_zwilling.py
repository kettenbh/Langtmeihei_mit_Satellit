"""Zwillingsexperiment: Schnittzeitpunkt und Futterqualität mit und ohne Satellit (ohne Internet).

    python scripts/qualitaet_zwilling.py

Wahrheit je Feld wie in scripts/relativ_zwilling.py: andere Parameter, eigenes Lokalwetter; der Landwirt mäht
an den Regelterminen seines Feldwetters, verschoben um bis zu ±10 Tage. Daraus eine synthetische S2-Reihe. Zu drei Stichtagen (10. Mai, 25. Juni, 10. August) rechnen
  Modell:  App-Parameter, Wetterpunkt, Schnittregel bis zum Stichtag (heutiges Verfahren)
  Hybrid:  EnKF-Analyse mit S2-LAI und erkannten Schnitten bis zum Stichtag (Varianten: hyb_alt = erster Stand
           ohne Blattalter-Korrektur, mit Resampling; hyb_nach = mit Blattalter und Nachtragen; hybrid = zusätzlich
           Schnitt-Hypothesen aus der Schnittregel; mix = hybrid, NEL/XP gemittelt mit dem Modell, so in der App)
die Kurve Menge/Qualität der nächsten 40 Tage über alle Klimajahre. Die Wahrheit läuft mit ihrem echten
Wetter ohne Schnitt weiter. Verglichen werden Median-Vorhersage und Wahrheit:
  - Heu netto, NEL und Rohprotein bei Schnitt in 14 Tagen,
  - Tag des maximalen Energieertrags.
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
JAHRE = list(range(2019, 2026))
STICHTAGE = {"10. Mai": 129, "25. Juni": 175, "10. Aug.": 221}
H = 14
VARIANTEN = {
    "hyb_alt": dict(alter_korrigieren=False, schnitt_resampling=30, schnitt_nachtragen=False, regel_hypothesen=False),
    "hyb_nach": dict(regel_hypothesen=False),
    "hybrid": dict(),
}


def wahre_kurve(S, P, wf, stichtag):
    """Wahrheit ab Stichtag mit echtem Feldwetter, ohne Schnitt: Tageswerte netto, NEL, XP, Energie."""
    clim1 = {0: wf}
    return Q._kurve_aus_ensemble(S, P, wf, clim1, stichtag, A, 40, (0.0, 0.0))


df = synthetic_weather(1990, dt.date(2025, 12, 31), seed=11)
clim = {y: year_arrays(df, y) for y in range(1991, 2019)}
p_app = ModVegeParams(NI=0.70, WHC=A.nutzbare_feldkapazitaet_mm)
rows = []
for k in range(FELDER):
    rng = np.random.default_rng(300 + k)
    p_w = Z.wahre_parameter(p_app, NI=rng.uniform(0.5, 0.9), WHC=rng.uniform(70, 160), RUEmax=rng.uniform(2.6, 3.4))
    for y in JAHRE:
        w = year_arrays(df, y)
        wf = feldwetter(w, rng)
        Pw = E.EnsembleParams.from_params(p_w, 1)
        # Der Landwirt mäht nicht nach der Schnittregel der App: Regeltermine (mit Feldwetter) um bis zu
        # ±10 Tage verschoben (Arbeitsspitzen, eigene Einschätzung).
        s_r = E.run_ensemble(wf, Pw, A, manage=True)
        tage = [int(t) + int(rng.integers(-10, 11)) for t in s_r.cut_day[0] if t >= 0]
        fc = np.array(sorted(tage))[None, :] if tage else None
        _, tr = E.run_ensemble(wf, Pw, A, manage=False, forced_cuts=fc, trace=True)
        s2 = Z.synthetische_s2_reihe(tr["lai"][:, 0], y, rng)
        cuts = SAT.detect_cuts(s2)
        tab = B.aus_s2(s2, list(cuts["datum"]) if len(cuts) else None)
        jd = R.JahrDaten(w, 365, B.jahr(tab, y), R.fenster_aus_schnitten(cuts, y))
        for name, t in STICHTAGE.items():
            Sw = E.run_ensemble(wf, Pw, A, end=t + 1, manage=False, forced_cuts=fc)
            kw = wahre_kurve(Sw, Pw, wf, t)
            km = Q.schnittkurve_modell_ensemble(w, t, clim, p_app, A, horizon=40)
            kurven = {"modell": km}
            for lab, kw_ in VARIANTEN.items():
                kurven[lab], _ = Q.schnittkurve_hybrid(jd, t, clim, p_app, A, horizon=40, n=24, seed=k * 1000 + y, **kw_)
            kurven["mix"] = Q.kurve_mischen(kurven["hybrid"], km)
            if kw.empty:
                continue
            tw = kw.set_index("tag")
            r = {"feld": k, "jahr": y, "stichtag": name, "best_wahr": int(tw["energie_gj_q50"].idxmax())}
            for lab, kk in kurven.items():
                kk = kk.set_index("tag")
                if kk.empty:
                    continue
                r[f"best_{lab}"] = int(kk["energie_gj_q50"].idxmax())
                if t + H in kk.index and t + H in tw.index:
                    for g in ("netto_dt", "nel", "xp"):
                        r[f"{g}_{lab}"] = kk.loc[t + H, f"{g}_q50"]
                        r[f"{g}_wahr"] = tw.loc[t + H, f"{g}_q50"]
            rows.append(r)
d = pd.DataFrame(rows)
print(f"{FELDER} Felder × {len(JAHRE)} Jahre, mittlerer absoluter Fehler der Median-Vorhersage\n")
print(f"{'Stichtag':9s} {'':8s} {'Heu netto in 14 T [dt]':>23s} {'NEL [MJ]':>9s} {'XP [g]':>7s} {'bester Tag [Tage]':>18s}")
for name in STICHTAGE:
    x = d[d["stichtag"] == name]
    for lab in ["modell", *VARIANTEN, "mix"]:
        e = lambda g: (x[f"{g}_{lab}"] - x[f"{g}_wahr"]).abs().mean()
        print(f"{name:9s} {lab:8s} {e('netto_dt'):>23.1f} {e('nel'):>9.2f} {e('xp'):>7.0f} "
              f"{(x[f'best_{lab}'] - x['best_wahr']).abs().mean():>18.1f}")
