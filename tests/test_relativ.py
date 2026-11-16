"""Relative Ertragsschätzung: Hilfsfunktionen und ein kleiner Lauf mit Kunstdaten."""
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HEU_SYNTHETIC", "1")

from heu import ensemble as E  # noqa: E402
from heu import forecast as F  # noqa: E402
from heu import relativ as R  # noqa: E402
from heu import zwilling as Z  # noqa: E402
from heu.beobachtung import Beobachtungen  # noqa: E402
from heu.config import ANNAHMEN as A  # noqa: E402
from heu.model import ModVegeParams  # noqa: E402
from heu.weather import synthetic_weather, year_arrays  # noqa: E402


def test_relativ_loo():
    r = R.relativ_loo({2020: 100.0, 2021: 50.0, 2022: 150.0})
    assert r[2020] == 100.0 and r[2021] == 40.0 and r[2022] == 200.0


def test_lichtindex_waechst_mit_lai_und_kuerzt_skalierung():
    w = {"rad": np.full(365, 15.0)}
    tag = np.arange(80, 320, 10)
    o1 = Beobachtungen(tag, np.full(len(tag), 1.0), np.ones(len(tag)), np.ones(len(tag)))
    o3 = Beobachtungen(tag, np.full(len(tag), 3.0), np.ones(len(tag)), np.ones(len(tag)))
    assert R.lichtindex(o3, w) > R.lichtindex(o1, w) > 0
    assert np.isclose(R.lichtindex(o3, w, lai_a=1.5), R.lichtindex(
        Beobachtungen(tag, np.full(len(tag), 2.0), np.ones(len(tag)), np.ones(len(tag))), w))


def test_prognose_relativ_band_und_verengung():
    df = synthetic_weather(2015, dt.date(2021, 12, 31), seed=2)
    p = ModVegeParams(WHC=130.0)
    clim = {y: year_arrays(df, y) for y in range(2015, 2021)}
    rng = np.random.default_rng(0)
    teile = []
    for y in range(2018, 2022):
        _, tr = E.run_ensemble(year_arrays(df, y), E.EnsembleParams.from_params(p, 1), A, manage=True, trace=True)
        teile.append(Z.synthetische_s2_reihe(tr["lai"][:, 0], y, rng))
    s2 = pd.concat(teile)
    daten = R.jahr_daten(df, s2, None, range(2018, 2022), clim)
    ep = R.feste_mitglieder(p, A, 12)
    fr = R.prognose_relativ(2021, daten, [2018, 2019, 2020], clim, 100, p, A, ep)
    sp = R.prognose_relativ(2021, daten, [2018, 2019, 2020], clim, 320, p, A, ep)
    assert fr.prozent.shape == (12 * len(clim),) and np.all(np.isfinite(fr.prozent))
    # Ende der Saison: kein Restwetter mehr, alle Szenarien eines Mitglieds gleich
    assert np.allclose(sp.prozent.reshape(12, -1).std(axis=1), 0)
    assert 50 < np.median(fr.prozent) < 160 and 50 < np.median(sp.prozent) < 160
