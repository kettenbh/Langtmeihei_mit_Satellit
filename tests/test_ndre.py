"""N-Status aus NDRE: Referenzkurve, Faktor, Wirkung auf das Rohprotein."""
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HEU_SYNTHETIC", "1")

from heu import ndre as ND  # noqa: E402
from heu import zwilling as Z  # noqa: E402


def _reihe(ni_rel=None, jahre=range(2018, 2024), seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(365)
    lai = np.clip(5 * np.sin(np.pi * (t - 80) / 230), 0.8, None)   # nie offener Boden (sonst "Acker")
    return pd.concat([Z.synthetische_s2_reihe(lai, y, rng, dunst=0.0, ni_rel=ni_rel) for y in jahre])


def test_ci_umkehrbar():
    x = np.array([0.2, 0.5, 0.7])
    assert np.allclose(ND.ndre_aus_ci(ND.ci_re(x)), x)


def test_referenz_trifft_erzeugende_kurve():
    ref = ND.referenz(_reihe())
    a, b, c = Z.CI_REF_NIEDERHAM
    lai = np.array([1.5, 3.0, 5.0])
    assert np.allclose(ref(lai), a * (1 - np.exp(-b * lai)) + c, rtol=0.08)
    assert 0.06 < ref.sigma_log < 0.16


def test_n_faktor_folgt_n_status_und_schrumpft():
    ref = ND.referenz(_reihe())
    blass = _reihe(ni_rel=np.full(365, 0.8), jahre=[2024], seed=5)
    st = ND.n_status(blass, ref)
    nf = ND.n_faktor(st, pd.Timestamp(2024, 7, 1))
    assert nf.n >= 1 and 0.75 < nf.faktor < 0.98       # Richtung stimmt, zur 1 hin geschrumpft
    assert nf.faktor >= nf.r_mittel - 1e-9 or nf.r_mittel > 0.98
    # ohne Aufnahme im Fenster: Faktor 1
    assert ND.n_faktor(st, pd.Timestamp(2024, 2, 1)).faktor == 1.0


def test_letzter_schnitt_begrenzt_fenster():
    ref = ND.referenz(_reihe())
    st = ND.n_status(_reihe(jahre=[2024], seed=7), ref)
    t = pd.Timestamp(2024, 7, 1)
    nf = ND.n_faktor(st, t, letzter_schnitt=t - pd.Timedelta(days=10))
    assert nf.n == 0 or nf.von > t - pd.Timedelta(days=4)
