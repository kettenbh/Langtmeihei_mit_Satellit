"""Schnitterkennung gegen bekannte Schnitttage prüfen (synthetisch, ohne Internet).

    python scripts/schnitt_eval.py                 # neue Erkennung
    python scripts/schnitt_eval.py --alt PFAD.py   # zusätzlich eine alte satellite.py zum Vergleich

Wahrheit: ModVege mit Schnittregel auf synthetischem Wetter (2017–2025, mehrere Seeds). Daraus eine
S2-ähnliche Reihe: Überflug alle 5 Tage, 55 % bewölkt, NDVI = 0,91 − 0,70·exp(−0,96·LAI) + N(0; 0,035)
(an 406 klaren Aufnahmen von Wiese Ost und West angepasst, Reststreuung 0,038),
LAI mit 25 % relativem Fehler, 5 % der klaren Aufnahmen als unerkannter Dunst (NDVI −0,15…−0,45).
Treffer: erkannter Schnitt höchstens 10 Tage neben einem wahren.
"""
import datetime as dt
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HEU_SYNTHETIC", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from heu import satellite as SAT  # noqa: E402
from heu import zwilling as Z  # noqa: E402
from heu.config import ANNAHMEN as A  # noqa: E402
from heu.model import ModVegeParams  # noqa: E402
from heu.weather import synthetic_weather, year_arrays  # noqa: E402

TOL = 10


s2_reihe = Z.synthetische_s2_reihe


def bewerte(detect, faelle):
    tp = fp = fn = 0
    fehler = []
    for s2, wahr in faelle:
        cuts = detect(s2)
        erk = [int(d.day_of_year) - 1 for d in cuts["datum"]] if len(cuts) else []
        wahr = [t for t in wahr if 90 <= t <= 305]          # April bis Oktober
        frei = list(wahr)
        for e in erk:
            m = min(frei, key=lambda t: abs(t - e)) if frei else None
            if m is not None and abs(m - e) <= TOL:
                tp += 1
                fehler.append(abs(m - e))
                frei.remove(m)
            else:
                fp += 1
        fn += len(frei)
    return tp, fp, fn, (float(np.mean(fehler)) if fehler else np.nan)


def main():
    df = synthetic_weather(2015, dt.date(2025, 12, 31), seed=11)
    p = ModVegeParams(NI=0.70, WHC=A.nutzbare_feldkapazitaet_mm)
    faelle = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        for jahr in range(2017, 2026):
            w = year_arrays(df, jahr)
            pw = Z.wahre_parameter(p, NI=float(rng.uniform(0.55, 0.85)))
            wahr = Z.wahrheit(w, pw, A)
            faelle.append((s2_reihe(wahr.trace["lai"][:, 0], jahr, rng), wahr.schnitt_tage))
    n_wahr = sum(len([t for t in w if 90 <= t <= 305]) for _, w in faelle)
    print(f"{len(faelle)} Jahre, {n_wahr} wahre Schnitte (April–Oktober), Toleranz ±{TOL} Tage\n")
    varianten = [("neu", SAT.detect_cuts)]
    if "--alt" in sys.argv:
        spec = importlib.util.spec_from_file_location("heu._sat_alt", sys.argv[sys.argv.index("--alt") + 1])
        alt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(alt)
        varianten.insert(0, ("alt", alt.detect_cuts))
    print(f"{'':5} {'Treffer':>7} {'Fehlalarm':>9} {'verpasst':>8} {'Trefferquote':>12} {'Präzision':>9} {'Ø Abw.':>6}")
    for name, det in varianten:
        tp, fp, fn, mae = bewerte(det, faelle)
        print(f"{name:5} {tp:>7} {fp:>9} {fn:>8} {tp / max(1, tp + fn):>12.0%} {tp / max(1, tp + fp):>9.0%} {mae:>6.1f}")


if __name__ == "__main__":
    main()
