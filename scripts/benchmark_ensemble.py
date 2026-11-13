"""Laufzeitvergleich Skalar- gegen Ensemble-ModVege (ohne Internet).

    python scripts/benchmark_ensemble.py

Szenario wie die geplante Prognose (Plan 3.5): 64 Analyse-Mitglieder × 35 Wetterjahre ab Stichtag.
"""
import datetime as dt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HEU_SYNTHETIC", "1")

import numpy as np  # noqa: E402

from heu import ensemble as E  # noqa: E402
from heu import model as M  # noqa: E402
from heu.config import ANNAHMEN as A  # noqa: E402
from heu.weather import synthetic_weather, year_arrays  # noqa: E402

N_MEMBERS, STICHTAG = 64, 120
df = synthetic_weather(1991, dt.date(2026, 6, 30), seed=1)
clim = {y: year_arrays(df, y) for y in range(1991, 2026)}
ist = year_arrays(df, 2024)
p = M.ModVegeParams(NI=0.7)
rng = np.random.default_rng(0)
ep = E.perturb_params(p, N_MEMBERS, rng, {"NI": 0.15, "RUEmax": 0.1})

# Zustand am Stichtag je Mitglied (Analysephase, hier ohne Assimilation)
es0 = E.run_ensemble(ist, ep, A, end=STICHTAG + 1)

years = sorted(clim)
w_szen = [{k: np.concatenate([ist[k][: STICHTAG + 1], clim[y][k][STICHTAG + 1:]]) for k in ist} for y in years]

# --- Skalar: jede Kombination einzeln ---------------------------------------
t = time.perf_counter()
states0 = es0.to_states()
tot_scalar = []
for i in range(N_MEMBERS):
    pi = ep.member(i)
    for w in w_szen:
        s = M.run(w, pi, A, state=states0[i].copy(), start=STICHTAG + 1, end=365)
        tot_scalar.append(M.season_total(s))
t_scalar = time.perf_counter() - t

# --- Ensemble: alle Kombinationen in einem Lauf ----------------------------
t = time.perf_counter()
k = len(years)
es = es0.repeat(k)                  # Mitglied i, Jahr j  ->  Zeile i*k + j
pp = ep.repeat(k)
ww = E.stack_weather(w_szen * N_MEMBERS)
E.run_ensemble(ww, pp, A, state=es, start=STICHTAG + 1, end=365)
tot_ens = es.season_total()
t_ens = time.perf_counter() - t

n = N_MEMBERS * k
diff = np.max(np.abs(np.array(tot_scalar) - tot_ens))
print(f"{n} Läufe ({N_MEMBERS} Mitglieder × {k} Wetterjahre), Stichtag Tag {STICHTAG}")
print(f"  Skalar:   {t_scalar:6.2f} s")
print(f"  Ensemble: {t_ens:6.2f} s   (Faktor {t_scalar / t_ens:.0f})")
print(f"  größte Abweichung Saisonertrag: {diff:.2e} kg TM/ha")
q = np.percentile(tot_ens / 100, [10, 50, 90])
print(f"  Saisonertrag netto q10/q50/q90: {q.round(1)} dt TM/ha")
