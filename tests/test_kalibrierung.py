"""ES-MDA-Kalibrierung im Zwillingsexperiment: Wahrheit mit NI 0,5 und WHC 90, Prior um 0,7 / 130."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HEU_SYNTHETIC", "1")

import datetime as dt  # noqa: E402

from heu import kalibrierung as K  # noqa: E402
from heu import zwilling as Z  # noqa: E402
from heu.config import ANNAHMEN as A  # noqa: E402
from heu.model import ModVegeParams  # noqa: E402
from heu.weather import synthetic_weather, year_arrays  # noqa: E402


def _daten(jahre, p_wahr):
    df = synthetic_weather(2016, dt.date(2021, 12, 31), seed=4)
    w, obs, fen, ertrag = {}, {}, {}, []
    for y in jahre:
        w[y] = year_arrays(df, y)
        wahr = Z.wahrheit(w[y], p_wahr, A)
        obs[y] = Z.synthetische_lai(wahr.trace["lai"][:, 0], np.random.default_rng(y))
        fen[y] = [(t - 4, t + 4) for t in wahr.schnitt_tage]
        ertrag.append(float(wahr.zustand.season_total(netto=False)[0]))
    return K.KalibrierDaten(list(jahre), w, obs, fen), float(np.mean(ertrag))


def test_esmda_lernt_parameter_und_schnitttage():
    p0 = ModVegeParams(WHC=130.0)
    p_wahr = Z.wahre_parameter(p0, NI=0.5, WHC=90.0)
    d, ertrag = _daten(range(2017, 2020), p_wahr)
    a = A.__class__(**{**A.__dict__, "ziel_jahresertrag_dt_tm_ha": ertrag / 100})
    erg = K.esmda(d, a, n=64, seed=1, namen=("NI", "RUEmax", "WHC", "lai_a"), p0=p0)
    post = dict(zip(erg.namen, np.median(erg.posterior, axis=0)))
    assert abs(post["NI"] - 0.5) < abs(0.7 - 0.5)          # näher an der Wahrheit als der Prior
    assert erg.lai_rmse["posterior"] < erg.lai_rmse["prior"]
    assert 0.8 < post["lai_a"] < 1.25                      # Operator ohne Verzerrung bleibt nahe 1
    # geschätzte Schnitttage liegen im Fenster
    for y, fen in d.schnitt_fenster.items():
        for (f, s), (med, lo, hi) in zip(fen, erg.schnitte[y]):
            assert f <= lo <= med <= hi <= s


def test_speichern_und_laden(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "DATA_DIR", tmp_path)
    p0 = ModVegeParams(WHC=130.0)
    d, _ = _daten([2018], p0)
    erg = K.esmda(d, A, n=16, seed=0, alphas=(1.0,), p0=p0)
    K.speichern(erg, "testfeld")
    ep, lai_a = K.laden("testfeld", A)
    assert ep.n == 16 and np.allclose(ep.NI, erg.posterior[:, 0]) and len(lai_a) == 16
    assert K.laden("gibtsnicht", A) is None
