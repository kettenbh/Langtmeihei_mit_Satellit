"""Zwillingsexperiment, Stand Paket 1: Open-Loop-Messlatte ohne Assimilation (ohne Internet).

    python scripts/twin_experiment.py

Wahrheit: Feld mit schlechterer Nährstoffversorgung (NI 0,55 statt 0,70) und geringerer
Wasserspeicherung (WHC 100 statt 130 mm). Die App kennt nur die Standardannahmen.
Ausgegeben werden LAI-Fehler, Streuung und Saisonertrag des Prior-Ensembles. Paket 3 (EnKF) muss
diese Werte schlagen; das Skript wird dann um die Spalte "EnKF" ergänzt.
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HEU_SYNTHETIC", "1")

import numpy as np  # noqa: E402

from heu import zwilling as Z  # noqa: E402
from heu.config import ANNAHMEN as A  # noqa: E402
from heu.model import ModVegeParams  # noqa: E402
from heu.weather import synthetic_weather, year_arrays  # noqa: E402

df = synthetic_weather(2015, dt.date(2025, 12, 31), seed=11)
p_app = ModVegeParams(NI=0.70, WHC=A.nutzbare_feldkapazitaet_mm)
p_wahr = Z.wahre_parameter(p_app, NI=0.55, WHC=100.0)

zeilen = []
for jahr in range(2017, 2026):
    w = year_arrays(df, jahr)
    wahr = Z.wahrheit(w, p_wahr, A)
    obs = Z.synthetische_lai(wahr.trace["lai"][:, 0], np.random.default_rng(jahr))
    s, tr = Z.open_loop(w, wahr, p_app, A, n=64, seed=jahr)
    ertrag_wahr = float(wahr.zustand.season_total()[0])
    b = Z.bewertung(tr["lai"], wahr.trace["lai"][:, 0], s.season_total(), ertrag_wahr, slice(60, 300))
    b["jahr"], b["n_obs"], b["schnitte"] = jahr, len(obs.tag), len(wahr.schnitt_tage)
    zeilen.append(b)

print("Open Loop (ohne Satellitenkorrektur), 64 Mitglieder, Wahrheit NI=0,55 / WHC=100 mm")
print(f"{'Jahr':>5} {'Obs':>4} {'Schn':>4} {'LAI-RMSE':>9} {'Spread':>7} {'Ertrag wahr':>12} {'Ens-Mittel':>11} {'CRPS':>7} {'Rang':>5}")
for b in zeilen:
    print(f"{b['jahr']:>5} {b['n_obs']:>4} {b['schnitte']:>4} {b['lai_rmse']:>9.2f} {b['lai_spread']:>7.2f} "
          f"{b['ertrag_wahr'] / 100:>12.1f} {b['ertrag_mittel'] / 100:>11.1f} {b['ertrag_crps'] / 100:>7.1f} {b['rang_wahrheit']:>5.2f}")
m = {k: np.mean([b[k] for b in zeilen]) for k in ("lai_rmse", "lai_spread", "ertrag_fehler", "ertrag_crps")}
print(f"Mittel: LAI-RMSE {m['lai_rmse']:.2f}, Spread {m['lai_spread']:.2f}, "
      f"Ertragsfehler {m['ertrag_fehler'] / 100:+.1f} dt/ha, CRPS {m['ertrag_crps'] / 100:.1f} dt/ha")
print("Erwartung: Open Loop überschätzt den Ertrag deutlich (falsches NI). Mit EnKF (Paket 3) sollen")
print("Ertragsfehler und CRPS sinken und der Rang der Wahrheit Richtung 0,5 wandern.")
