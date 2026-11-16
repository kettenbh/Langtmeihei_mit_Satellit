"""Hybrid-Schnittkurve (Futterqualität mit Sentinel-2): Schnitt-Hypothesen, Nachtragen, Blattalter."""
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HEU_SYNTHETIC", "1")

from heu import assimilation as DA_  # noqa: E402
from heu import ensemble as E  # noqa: E402
from heu import qualitaet as Q  # noqa: E402
from heu import relativ as R  # noqa: E402
from heu.beobachtung import Beobachtungen  # noqa: E402
from heu.config import ANNAHMEN as A  # noqa: E402
from heu.model import ModVegeParams  # noqa: E402
from heu.weather import synthetic_weather, year_arrays  # noqa: E402

DF = synthetic_weather(2015, dt.date(2021, 12, 31), seed=2)
P = ModVegeParams(WHC=130.0)
W = year_arrays(DF, 2021)
CLIM = {y: year_arrays(DF, y) for y in range(2016, 2021)}


def _regel_schnitte(stichtag):
    S = E.run_ensemble({k: v[None, :] for k, v in W.items()}, E.EnsembleParams.from_params(P, 1), A,
                       end=stichtag + 1, manage=True)
    return [int(c) for c in S.cut_day[0] if 0 <= c <= stichtag]


def test_regel_hypothesen_nur_ohne_erkannten_schnitt():
    c = _regel_schnitte(200)
    assert c, "Kunstwetter sollte bis Juli mindestens einen Regelschnitt haben"
    h = Q.regel_schnitt_hypothesen(W, P, A, 200, [])
    assert [(f + 7) for f, _ in h] == c
    h2 = Q.regel_schnitt_hypothesen(W, P, A, 200, [(c[0] - 3, c[0] + 3)])
    assert len(h2) == len(c) - 1 and all(f != c[0] - 7 for f, _ in h2)


def test_hypothese_nur_bei_einem_teil_der_mitglieder():
    an = DA_.analyse_saison(W, P, A, None, None, ende=200, n=40, seed=1, schnitt_hypothesen=[(150, 160)],
                            alter_korrigieren=True)
    hat = (an.zustand.cut_day >= 150).any(axis=1)
    assert 5 < hat.sum() < 35


def test_lai_einbruch_traegt_schnitt_nach_und_verjuengt():
    # hoher LAI bis Tag 150, dann LAI 0,8 an Tag 165: ein Schnitt dazwischen, ohne erkanntes Fenster
    tag = np.array([110, 125, 140, 150, 165, 175])
    lai = np.array([2.0, 4.0, 5.5, 6.0, 0.8, 1.6])
    obs = Beobachtungen(tag, lai, np.full(6, 0.4), np.ones(6))
    an = DA_.analyse_saison(W, P, A, obs, None, ende=176, n=24, seed=3, schnitt_fenster=[],
                            schnitt_nachtragen=True, alter_korrigieren=True)
    assert an.nachgetragene_schnitte == [(151, 165)]
    assert ((an.zustand.last_cut > 150) & (an.zustand.last_cut <= 165)).all()
    ohne = DA_.analyse_saison(W, P, A, obs, None, ende=176, n=24, seed=3, schnitt_fenster=[])
    assert np.median(an.zustand.aGV) < np.median(ohne.zustand.aGV)


def test_schnittkurve_hybrid_liefert_kurve_und_info():
    tag = np.arange(100, 181, 10)
    obs = Beobachtungen(tag, np.linspace(1.5, 4.0, len(tag)), np.full(len(tag), 0.5), np.ones(len(tag)))
    jd = R.JahrDaten(W, 365, obs, [(150, 156)])
    k, info = Q.schnittkurve_hybrid(jd, 180, CLIM, P, A, horizon=20, n=12)
    assert not k.empty and {"netto_dt_q50", "nel_q50", "xp_q50", "energie_gj_q50"} <= set(k.columns)
    assert info["aufnahmen"] == len(tag) and info["schnitte"] == [(150, 156)]
    assert k["nel_q50"].between(3, 8).all()
