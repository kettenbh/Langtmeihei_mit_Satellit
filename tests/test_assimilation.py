"""EnKF-Baustein, Transformationen, Beobachtungs-QC und Zwillingsexperiment im Kleinen."""
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HEU_SYNTHETIC", "1")

from heu import assimilation as DA  # noqa: E402
from heu import beobachtung as B  # noqa: E402
from heu import ensemble as E  # noqa: E402
from heu import zwilling as Z  # noqa: E402
from heu.config import ANNAHMEN as A, DA as CFG  # noqa: E402
from heu.model import ModVegeParams  # noqa: E402
from heu.weather import synthetic_weather, year_arrays  # noqa: E402


def test_enkf_linear_gaussian_matches_kalman():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40000, 1))
    Xa, used, _ = DA.enkf_update(X, X[:, 0].copy(), y=1.0, r=1.0, rng=rng)
    assert used
    assert abs(Xa.mean() - 0.5) < 0.02          # Kalman: Mittel 0,5
    assert abs(Xa.var() - 0.5) < 0.03           # Kalman: Varianz 0,5


def test_enkf_gating_rejects_outlier():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (500, 2))
    Xa, used, info = DA.enkf_update(X, X[:, 0].copy(), y=50.0, r=1.0, rng=rng, gating=3.0)
    assert not used and info["grund"] == "Gating"
    np.testing.assert_allclose(Xa, X)


def test_enkf_damping_scales_increment():
    X = np.random.default_rng(2).normal(0, 1, (5000, 2))
    X[:, 1] = X[:, 0]                            # zweite Komponente voll korreliert
    a, _, _ = DA.enkf_update(X.copy(), X[:, 0].copy(), 1.0, 1.0, np.random.default_rng(3))
    b, _, _ = DA.enkf_update(X.copy(), X[:, 0].copy(), 1.0, 1.0, np.random.default_rng(3), damp=np.array([1.0, 0.4]))
    np.testing.assert_allclose(b[:, 0], a[:, 0])
    np.testing.assert_allclose(b[:, 1] - X[:, 1], 0.4 * (a[:, 1] - X[:, 1]), atol=1e-12)


def test_vector_roundtrip():
    p = ModVegeParams()
    ep = E.perturb_params(p, 10, np.random.default_rng(0), {"NI": 0.2, "RUEmax": 0.1})
    s = E.perturb_state(E.EnsembleState.initial(ep), np.random.default_rng(1))
    s.GR[:] = 0.0
    s.WR = ep.WHC * 0.4
    ref = s.copy()
    X = DA.to_vector(s, ep, ("NI", "RUEmax"))
    ni = ep.NI.copy()
    DA.from_vector(X, s, ep, ("NI", "RUEmax"), CFG.param_grenzen)
    for k in ("GV", "GR", "DV", "DR", "WR"):
        np.testing.assert_allclose(getattr(s, k), getattr(ref, k), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ep.NI, ni, rtol=1e-12)


def test_beobachtung_qc():
    idx = pd.to_datetime(["2024-01-15", "2024-04-01", "2024-04-06", "2024-04-11", "2024-04-16",
                          "2024-04-21", "2024-05-20", "2024-05-22", "2024-06-01"])
    s2 = pd.DataFrame({"lai": [0.5, 1.0, 1.2, 6.5, 1.5, 1.6, 3.0, 0.4, 0.8],
                       "clear": [1, 1, 0.5, 1, 1, 1, 1, 1, 1]}, index=idx)
    tab = B.aus_s2(s2, schnitte=[dt.date(2024, 5, 21)])
    g = dict(zip(tab["datum"].dt.strftime("%m-%d"), tab["grund"]))
    assert g["01-15"] == "Winter"
    assert g["04-06"] == "bewölkt"
    assert g["04-11"] == "Ausreißer"
    assert g["05-22"] == "frisch gemäht"
    assert g["04-16"] == "" and g["06-01"] == ""
    assert (tab["sigma"] >= 0.5).all()
    obs = B.jahr(tab, 2024)
    assert list(obs.tag) == sorted(obs.tag) and len(obs) == 5


def test_twin_enkf_beats_open_loop():
    """Kleines Zwillingsexperiment: der EnKF muss Ertrag und LAI deutlich besser treffen."""
    df = synthetic_weather(2015, dt.date(2020, 12, 31), seed=11)
    p_app = ModVegeParams(NI=0.70, WHC=A.nutzbare_feldkapazitaet_mm)
    p_wahr = Z.wahre_parameter(p_app, NI=0.55, WHC=100.0, RUEmax=3.2)
    err = {"ol": [], "enkf": []}
    lai = {"ol": [], "enkf": []}
    for jahr in (2017, 2018, 2019):
        w = year_arrays(df, jahr)
        wahr = Z.wahrheit(w, p_wahr, A)
        obs = Z.synthetische_lai(wahr.trace["lai"][:, 0], np.random.default_rng(jahr))
        y = wahr.zustand.season_total()[0]
        for k, assim in (("ol", False), ("enkf", True)):
            an = DA.analyse_saison(w, p_app, A, obs, wahr.schnitt_tage, seed=jahr, n=32,
                                   schnitt_unsicherheit=3, assimilieren=assim)
            err[k].append(abs(an.zustand.season_total().mean() - y))
            lai[k].append(np.sqrt(np.mean((an.trace["lai"][60:300].mean(axis=1) - wahr.trace["lai"][60:300, 0]) ** 2)))
    assert np.mean(err["enkf"]) < 0.5 * np.mean(err["ol"])
    assert np.mean(lai["enkf"]) < 0.8 * np.mean(lai["ol"])


def test_analysis_keeps_ensemble_valid():
    df = synthetic_weather(2015, dt.date(2018, 12, 31), seed=3)
    w = year_arrays(df, 2017)
    obs = B.Beobachtungen(np.array([100, 120, 140]), np.array([2.0, 3.5, 4.0]),
                          np.array([0.6, 0.9, 1.0]), np.ones(3))
    an = DA.analyse_saison(w, ModVegeParams(), A, obs, [150], ende=200, n=16)
    s = an.zustand
    for k in ("GV", "GR", "DV", "DR"):
        assert np.all(getattr(s, k) >= 0)
    assert np.all((s.WR >= 0) & (s.WR <= an.params.WHC))
    lo, hi = CFG.param_grenzen[0]
    assert np.all((an.params.NI >= lo) & (an.params.NI <= hi))
    assert an.trace["lai"].shape == (200, 16)
    assert len(an.updates) == 3 and len(an.param_verlauf["NI"][0]) == 3
