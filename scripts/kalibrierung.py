"""Stufe A: Kalibrierung eines Feldes mit ES-MDA (Plan Abschnitt 3.3). Läuft auf dem Mac mit echten Daten.

    python scripts/kalibrierung.py                       # Standardfeld (config.STANDARD_FELD)
    python scripts/kalibrierung.py --feld wiese_ost
    python scripts/kalibrierung.py --test 2024,2025      # diese Jahre zurückhalten und damit prüfen
    python scripts/kalibrierung.py -n 128

Voraussetzung: data/sat_s2_<feld>.csv (Satelliten-Tab der App) und data/wetter_cache.csv.
Ergebnis: data/kalibrierung_<feld>.csv (Posterior-Stichprobe, Prior für die Assimilation im neuen Jahr).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from heu import kalibrierung as K  # noqa: E402
from heu import satellite as SAT  # noqa: E402
from heu.config import ANNAHMEN as A, STANDARD_FELD  # noqa: E402
from heu.weather import load_weather  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--feld", default=STANDARD_FELD)
ap.add_argument("--von", type=int, default=2017)
ap.add_argument("--bis", type=int, default=2025)
ap.add_argument("--test", default="", help="Prüfjahre, z. B. 2024,2025 (nicht kalibriert)")
ap.add_argument("-n", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

s2 = SAT.load_cached(args.feld, "s2")
s1 = SAT.load_cached(args.feld, "s1")
if s2 is None or s2.empty:
    sys.exit(f"Keine Satellitendaten für {args.feld}: zuerst in der App abrufen.")
wetter, quelle = load_weather(1991)
cuts = SAT.detect_cuts(s2, s1)
test = [int(t) for t in args.test.split(",") if t]
jahre = [y for y in range(args.von, args.bis + 1) if y not in test]
d = K.daten(s2, cuts, wetter, jahre)
print(f"Feld {args.feld}, Wetter: {quelle}")
print(f"Kalibrierjahre: {d.jahre} ({d.n_obs} LAI-Aufnahmen, "
      f"{sum(len(v) for v in d.schnitt_fenster.values())} erkannte Schnitte)"
      + (f", Prüfjahre: {test}" if test else ""))
t0 = time.time()
erg = K.esmda(d, A, n=args.n, seed=args.seed, protokoll=lambda i, k: print(f"  Iteration {i}/{k} ({time.time() - t0:.0f} s)"))
print(f"\nAussortiert je Iteration (Gating): {erg.verworfen}")
print(f"LAI-RMSE Ensemblemittel gegen S2: Prior {erg.lai_rmse['prior']:.2f}, Posterior {erg.lai_rmse['posterior']:.2f}\n")
with np.printoptions(precision=3):
    print(erg.zusammenfassung().to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("\nBrutto-Jahresertrag [dt TM/ha], Median (10–90 %) im Posterior:")
for j, y in enumerate(erg.jahre):
    e = erg.ertrag["posterior"][:, j] / 100
    print(f"  {y}: {np.median(e):5.1f} ({np.percentile(e, 10):.1f}–{np.percentile(e, 90):.1f})")

if test:
    dt_ = K.daten(s2, cuts, wetter, test)
    rng = np.random.default_rng(args.seed + 1)
    sch = K._schnitt_ziehung(dt_, args.n, rng)
    y_obs = np.concatenate([dt_.obs[j].wert for j in dt_.jahre])
    print(f"\nPrüfjahre {dt_.jahre} ({dt_.n_obs} Aufnahmen, nicht in der Kalibrierung):")
    for name, X in (("Prior", erg.prior), ("Posterior", erg.posterior)):
        lai, _ = K.vorwaerts(X, erg.namen, dt_, A, sch)
        rmse = np.sqrt(np.mean((lai.mean(0) - y_obs) ** 2))
        bias = np.mean(lai.mean(0) - y_obs)
        print(f"  {name:9s}: LAI-RMSE {rmse:.2f}, Bias {bias:+.2f}")

if not test:
    print(f"\nGespeichert: {K.speichern(erg, args.feld)}")
else:
    print("\nMit --test wird nichts gespeichert (Endlauf ohne --test über alle Jahre).")
