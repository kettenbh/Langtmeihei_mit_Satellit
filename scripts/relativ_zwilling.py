"""Zwillingsexperiment für die relative Ertragsschätzung (% vom Mittel der übrigen Jahre), ohne Internet.

    python scripts/relativ_zwilling.py            # 8 Felder × 9 Jahre
    python scripts/relativ_zwilling.py --schnell  # 3 Felder

Je "Feld" eine Wahrheit, die systematisch anders ist als die Annahmen der App (NI 0,5–0,9, WHC 70–160 mm,
RUEmax 2,6–3,4) und lokal anderes Wetter hat als der Wetterpunkt (Regen je Jahr ×LN(0,3), je Tag ×LN(0,6),
Strahlung je Jahr ×LN(0,05)), wie bei einem Feld 7 km neben dem ERA5-Punkt. Die Wahrheit mäht nach der
Schnittregel mit ihrem eigenen Wetter. Daraus eine synthetische S2-Reihe (Wolken, Dunst, LAI-Fehler).

Verglichen werden die relativen Jahreswerte der Schätzer aus heu/relativ.py mit der relativen Wahrheit, einmal als
Erntesumme (hängt stark an Zahl und Lage der Schnitte) und einmal als Aufwuchs (Σ Bruttozuwachs April–Oktober).
Zweiter Durchgang mit Bestandesstörungen, die kein Wetter kennt (siehe stoerung()).
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
from heu import relativ as R  # noqa: E402
from heu import satellite as SAT  # noqa: E402
from heu import zwilling as Z  # noqa: E402
from heu.config import ANNAHMEN as A  # noqa: E402
from heu.model import ModVegeParams  # noqa: E402
from heu.weather import synthetic_weather, year_arrays  # noqa: E402

SCHNELL = "--schnell" in sys.argv
FELDER = 3 if SCHNELL else 8
JAHRE = list(range(2017, 2026))
N = 48


def feldwetter(w, rng):
    days = len(w["rain"])
    ln = lambda s, size: rng.lognormal(-0.5 * s * s, s, size)
    return {**w, "rain": w["rain"] * ln(0.3, 1) * ln(0.6, days), "rad": w["rad"] * ln(0.05, 1)}


def stoerung(rng, days=365):
    """Bestandesstörung, die kein Wetter kennt (Engerlinge, Mäuse, Narbenschaden, Hagel, Güllefehler):
    in 40 % der Jahre wächst der Bestand 30–70 Tage lang nur mit 40–75 % (gro_factor)."""
    f = np.ones(days)
    if rng.random() < 0.4:
        t0 = int(rng.integers(100, 240))
        f[t0:t0 + int(rng.integers(30, 71))] = rng.uniform(0.4, 0.75)
    return f


def ein_feld(k: int, df: pd.DataFrame, p_app: ModVegeParams, mit_stoerung: bool) -> pd.DataFrame:
    rng = np.random.default_rng(100 + k)
    p_w = Z.wahre_parameter(p_app, NI=rng.uniform(0.5, 0.9), WHC=rng.uniform(70, 160), RUEmax=rng.uniform(2.6, 3.4))
    wahr, wahr_z, s2_teile, w_punkt = {}, {}, [], {}
    for y in JAHRE:
        w = year_arrays(df, y)
        w_punkt[y] = w
        wf = feldwetter(w, rng)
        gf = stoerung(rng)[:, None] if mit_stoerung else None
        s, tr = E.run_ensemble(wf, E.EnsembleParams.from_params(p_w, 1), A, manage=True, trace=True, gro_factor=gf)
        wahr[y] = float(s.season_total(netto=False)[0])
        wahr_z[y] = float(tr["gro"][R.SAISON[0]:R.SAISON[1], 0].sum())
        s2_teile.append(Z.synthetische_s2_reihe(tr["lai"][:, 0], y, rng))
    s2 = pd.concat(s2_teile)
    cuts = SAT.detect_cuts(s2)
    tab = B.aus_s2(s2, list(cuts["datum"]) if len(cuts) else None)
    est = {k_: {} for k_ in ("wetter", "wetter_zuwachs", "hybrid_ernte", "hybrid_zuwachs", "licht")}
    for y in JAHRE:
        obs = B.jahr(tab, y)
        fen = R.fenster_aus_schnitten(cuts, y)
        s_w, tr_w = E.run_ensemble(w_punkt[y], E.EnsembleParams.from_params(p_app, 1), A, manage=True, trace=True)
        est["wetter"][y] = float(s_w.season_total(netto=False)[0])
        est["wetter_zuwachs"][y] = float(tr_w["gro"][R.SAISON[0]:R.SAISON[1], 0].sum())
        for g in ("ernte", "zuwachs"):
            est[f"hybrid_{g}"][y] = float(np.mean(R.ertrag_hybrid(w_punkt[y], p_app, A, obs, fen, n=N,
                                                                  seed=k * 100 + y, groesse=g)))
        est["licht"][y] = R.lichtindex(obs, w_punkt[y])
    rows = {"wahr_ernte": R.relativ_loo(wahr), "wahr_zuwachs": R.relativ_loo(wahr_z)}
    for name, d in est.items():
        rows[name] = R.relativ_loo(d)
    out = pd.DataFrame(rows)
    out["feld"] = k
    return out


def bericht(alle: pd.DataFrame, ziel: str):
    print(f"  gegen {ziel} (Jahr-zu-Jahr-Streuung der Wahrheit {alle[ziel].std():.1f} %-Pkt):")
    print(f"  {'Schätzer':16s} {'mittl. Fehler':>14s} {'Korrelation':>12s} {'größter Fehler':>15s}")
    alle = alle.assign(kombi=alle[["hybrid_zuwachs", "licht"]].mean(axis=1))
    for name in ("wetter", "wetter_zuwachs", "hybrid_ernte", "hybrid_zuwachs", "licht", "kombi"):
        d = alle[[name, ziel]].dropna()
        e = d[name] - d[ziel]
        print(f"  {name:16s} {e.abs().mean():>14.1f} {np.corrcoef(d[name], d[ziel])[0, 1]:>12.2f} {e.abs().max():>15.1f}")


if __name__ == "__main__":
    df = synthetic_weather(2015, dt.date(2025, 12, 31), seed=11)
    p_app = ModVegeParams(NI=0.70, WHC=A.nutzbare_feldkapazitaet_mm)
    for mit in (False, True):
        alle = pd.concat([ein_feld(k, df, p_app, mit) for k in range(FELDER)])
        print(f"\n=== {'mit Bestandesstörungen' if mit else 'nur Wetter- und Parameterunterschiede'}: "
              f"{FELDER} Felder × {len(JAHRE)} Jahre, % vom Mittel der übrigen Jahre ===")
        bericht(alle, "wahr_ernte")
        bericht(alle, "wahr_zuwachs")
        if "--detail" in sys.argv:
            print(alle.round(0).to_string())
