"""Zwillingsexperiment: EnKF gegen Open Loop (ohne Internet, Plan Abschnitt 4).

    python scripts/twin_experiment.py            # 9 Jahre, 64 Mitglieder
    python scripts/twin_experiment.py --schnell  # 3 Jahre, 32 Mitglieder

Wahrheit: Feld mit schlechterer Nährstoffversorgung (NI 0,55 statt 0,70), geringerer Wasserspeicherung
(WHC 100 statt 130 mm) und etwas höherer Strahlungsnutzung (RUEmax 3,2 statt 3,0). Die App kennt nur
die Standardannahmen. Beobachtet werden S2-LAI (alle 5 Tage, 55 % bewölkt, Fehler nach Plan 3.2) und
die Schnitttage (±3 Tage unsicher).

Drei Prüfungen:
  1. Saison-Rückblick: Analyse über das ganze Jahr, Jahresertrag gegen Wahrheit.
  2. Prognose am Stichtag 15. Mai: Analyse bis dahin, danach Schnittregel mit wahrem Wetter.
  3. Parameter: Was hat der Filter über NI gelernt?
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HEU_SYNTHETIC", "1")

import numpy as np  # noqa: E402

from heu import assimilation as DA  # noqa: E402
from heu import zwilling as Z  # noqa: E402
from heu.config import ANNAHMEN as A  # noqa: E402
from heu.model import ModVegeParams  # noqa: E402
from heu.weather import synthetic_weather, year_arrays  # noqa: E402

SCHNELL = "--schnell" in sys.argv
JAHRE = range(2017, 2020) if SCHNELL else range(2017, 2026)
N = 32 if SCHNELL else 64
STICHTAG = 134   # 15. Mai

df = synthetic_weather(2015, dt.date(2025, 12, 31), seed=11)
p_app = ModVegeParams(NI=0.70, WHC=A.nutzbare_feldkapazitaet_mm)
p_wahr = Z.wahre_parameter(p_app, NI=0.55, WHC=100.0, RUEmax=3.2)


def lauf(jahr: int) -> dict:
    w = year_arrays(df, jahr)
    wahr = Z.wahrheit(w, p_wahr, A)
    lai_w = wahr.trace["lai"][:, 0]
    obs = Z.synthetische_lai(lai_w, np.random.default_rng(jahr))
    y_w = float(wahr.zustand.season_total()[0])
    out = {"jahr": jahr, "n_obs": len(obs), "wahr": y_w}
    for name, assim in (("ol", False), ("enkf", True)):
        an = DA.analyse_saison(w, p_app, A, obs, wahr.schnitt_tage, seed=jahr, schnitt_unsicherheit=3,
                               assimilieren=assim, n=N)
        b = Z.bewertung(an.trace["lai"], lai_w, an.zustand.season_total(), y_w, slice(60, 300))
        out[f"{name}_lai"] = b["lai_rmse"]
        out[f"{name}_spread"] = b["lai_spread"]
        out[f"{name}_ertrag"] = b["ertrag_mittel"]
        out[f"{name}_crps"] = b["ertrag_crps"]
        out[f"{name}_ni"] = float(an.params.NI.mean())
        if assim:
            out["verworfen"] = sum(not u.verwendet for u in an.updates)
        # Prognose am Stichtag
        cuts_vor = [d for d in wahr.schnitt_tage if d <= STICHTAG]
        obs_vor = Z.Beobachtungen(obs.tag[obs.tag <= STICHTAG], obs.wert[obs.tag <= STICHTAG],
                                  obs.sigma[obs.tag <= STICHTAG], obs.clear[obs.tag <= STICHTAG])
        an_s = DA.analyse_saison(w, p_app, A, obs_vor, cuts_vor, ende=STICHTAG + 1, seed=jahr,
                                 schnitt_unsicherheit=3, assimilieren=assim, n=N)
        s_end = DA.prognose(an_s, w, A, start=STICHTAG + 1)
        tot = s_end.season_total()
        out[f"{name}_prog"] = float(tot.mean())
        out[f"{name}_prog_crps"] = Z.crps_ensemble(tot, y_w)
    return out


zeilen = [lauf(j) for j in JAHRE]
dt_ = lambda v: v / 100  # kg/ha -> dt/ha

print(f"Zwillingsexperiment: {N} Mitglieder, Wahrheit NI=0,55 WHC=100 RUEmax=3,2, App-Annahme NI=0,70 WHC=130 RUEmax=3,0\n")
print("1) Saison-Rückblick (Analyse über das ganze Jahr)")
print(f"{'Jahr':>5} {'Obs':>4} {'verw.':>5} {'wahr':>6} | {'OL Ertrag':>9} {'CRPS':>5} {'LAI-RMSE':>8} | {'EnKF Ertrag':>11} {'CRPS':>5} {'LAI-RMSE':>8} {'NI':>5}")
for z in zeilen:
    print(f"{z['jahr']:>5} {z['n_obs']:>4} {z['verworfen']:>5} {dt_(z['wahr']):>6.1f} | {dt_(z['ol_ertrag']):>9.1f} {dt_(z['ol_crps']):>5.1f} "
          f"{z['ol_lai']:>8.2f} | {dt_(z['enkf_ertrag']):>11.1f} {dt_(z['enkf_crps']):>5.1f} {z['enkf_lai']:>8.2f} {z['enkf_ni']:>5.2f}")


def mittel(k):
    return float(np.mean([z[k] for z in zeilen]))


def mae(k):
    return float(np.mean([abs(z[k] - z["wahr"]) for z in zeilen]))


print(f"Mittel  Open Loop: Ertragsfehler {dt_(mae('ol_ertrag')):.1f} dt/ha, CRPS {dt_(mittel('ol_crps')):.1f}, LAI-RMSE {mittel('ol_lai'):.2f}, Spread {mittel('ol_spread'):.2f}")
print(f"Mittel  EnKF:      Ertragsfehler {dt_(mae('enkf_ertrag')):.1f} dt/ha, CRPS {dt_(mittel('enkf_crps')):.1f}, LAI-RMSE {mittel('enkf_lai'):.2f}, Spread {mittel('enkf_spread'):.2f}")
print(f"        NI geschätzt: {mittel('enkf_ni'):.2f} (wahr 0,55, Prior 0,70)\n")
print("2) Prognose am 15. Mai (danach Schnittregel, wahres Wetter)")
print(f"Open Loop: Fehler {dt_(mae('ol_prog')):.1f} dt/ha, CRPS {dt_(mittel('ol_prog_crps')):.1f}")
print(f"EnKF:      Fehler {dt_(mae('enkf_prog')):.1f} dt/ha, CRPS {dt_(mittel('enkf_prog_crps')):.1f}")
