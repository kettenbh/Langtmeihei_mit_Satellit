"""Regressionstest: heu/ensemble.py muss dieselben Ergebnisse liefern wie die Skalarversion heu/model.py.

Lauf ohne Internet:  python -m pytest -q tests/
"""
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HEU_SYNTHETIC", "1")

from heu import ensemble as E  # noqa: E402
from heu import model as M  # noqa: E402
from heu.config import ANNAHMEN as A  # noqa: E402
from heu.weather import synthetic_weather, year_arrays  # noqa: E402
import datetime as dt  # noqa: E402

RTOL = 1e-9
FLOATS = ("GV", "GR", "DV", "DR", "aGV", "aGR", "aDV", "aDR", "WR", "ST")


@pytest.fixture(scope="module")
def years():
    df = synthetic_weather(2000, dt.date(2025, 12, 31), seed=7)
    return {y: year_arrays(df, y) for y in range(2000, 2026)}


@pytest.fixture(scope="module")
def p0():
    return M.ModVegeParams(NI=0.7, WHC=A.nutzbare_feldkapazitaet_mm)


def assert_state_equal(es: E.EnsembleState, i: int, s: M.State):
    for k in FLOATS:
        np.testing.assert_allclose(getattr(es, k)[i], getattr(s, k), rtol=RTOL, atol=1e-9, err_msg=k)
    assert bool(es.repro_done[i]) == s.repro_done
    assert int(es.n_cuts[i]) == s.n_cuts
    assert int(es.last_cut[i]) == s.last_cut
    assert int(es.ready_since[i]) == s.ready_since
    assert int(es.n_cuts[i]) == len(s.cuts)
    for j, c in enumerate(s.cuts):
        assert int(es.cut_day[i, j]) == c.day, f"Schnitt {j + 1}: Tag"
        np.testing.assert_allclose(es.cut_brutto[i, j], c.brutto_kg, rtol=RTOL, atol=1e-9)
        np.testing.assert_allclose(es.cut_netto[i, j], c.netto_kg, rtol=RTOL, atol=1e-9)
        np.testing.assert_allclose(es.cut_omd[i, j], c.omd, rtol=RTOL, atol=1e-12)
        assert int(es.cut_regentage[i, j]) == c.regentage
        assert bool(es.cut_erzwungen[i, j]) == c.erzwungen


def test_dry_windows_matches_scalar(years):
    w = years[2010]
    ok, rt = E.dry_windows(w["rain"], w["et0"], A)
    for d in range(365):
        o, r = M.dry_window(w, d, A)
        assert bool(ok[d]) == o and int(rt[d]) == r, d


def test_growth_only_identical(years, p0):
    """Ohne Schnitte: reiner Tagesschritt über ein ganzes Jahr."""
    w = years[2005]
    s = M.run(w, p0, A, manage=False)
    ep = E.EnsembleParams.from_params(p0, 3)
    es = E.run_ensemble(w, ep, A, manage=False)
    for i in range(3):
        assert_state_equal(es, i, s)


@pytest.mark.parametrize("year", [2001, 2008, 2013, 2018, 2024])
def test_full_season_with_cut_rule(years, p0, year):
    w = years[year]
    s = M.run(w, p0, A)
    es = E.run_ensemble(w, E.EnsembleParams.from_params(p0, 1), A)
    assert_state_equal(es, 0, s)
    np.testing.assert_allclose(es.season_total()[0], M.season_total(s), rtol=RTOL)


def test_member_specific_parameters(years, p0):
    """Jedes Mitglied eigene Parameter: Ergebnis = Skalarlauf mit genau diesen Parametern."""
    rng = np.random.default_rng(1)
    ep = E.perturb_params(p0, 12, rng, {"NI": 0.2, "RUEmax": 0.1, "WHC": 0.2, "ST1": 0.05, "LLS": 0.1})
    w = years[2016]
    es = E.run_ensemble(w, ep, A)
    for i in range(ep.n):
        s = M.run(w, ep.member(i), A)
        assert_state_equal(es, i, s)


def test_member_specific_weather(years, p0):
    """Jedes Mitglied ein anderes Klimajahr (Szenarien-Prognose)."""
    ys = sorted(years)
    ws = E.stack_weather([years[y] for y in ys])
    es = E.run_ensemble(ws, E.EnsembleParams.from_params(p0, len(ys)), A)
    for i, y in enumerate(ys):
        assert_state_equal(es, i, M.run(years[y], p0, A))


def test_restart_from_snapshot(years, p0):
    """Wie forecast(): Skalarzustand am Stichtag kopieren, dann mit Szenario-Wetter weiterrechnen."""
    w_ist, w_szen = years[2020], years[2011]
    stichtag = 170
    s0 = M.run(w_ist, p0, A, end=stichtag + 1)
    assert s0.cuts, "Testannahme: vor dem Stichtag hat es schon geschnitten"
    w = {k: np.concatenate([w_ist[k][: stichtag + 1], w_szen[k][stichtag + 1:]]) for k in w_ist}
    ref = M.run(w, p0, A, state=s0.copy(), start=stichtag + 1, end=365)
    es = E.EnsembleState.from_state(s0, 4)
    E.run_ensemble(w, E.EnsembleParams.from_params(p0, 4), A, state=es, start=stichtag + 1, end=365)
    for i in range(4):
        assert_state_equal(es, i, ref)


def test_forced_cuts_match_manual_scalar(years, p0):
    """Beobachtete Schnitttage: Ergebnis = Skalarmodell, das genau an diesen Tagen schneidet."""
    w = years[2019]
    days = [150, 197, 251]
    s = M.State(WR=p0.WHC)
    for d in range(365):
        if d in days:
            M.do_cut(s, d, p0, A, 0, False)
        M.grow_one_day(s, w["tmean"][d], w["rad"][d], w["rain"][d], w["et0"][d], p0)
    es = E.run_ensemble(w, E.EnsembleParams.from_params(p0, 2), A, manage=False, forced_cuts=days)
    for i in range(2):
        assert_state_equal(es, i, s)
    assert es.cut_beobachtet[:, :3].all()


def test_forced_cuts_per_member(years, p0):
    w = years[2019]
    fc = np.array([[150, 197, -1], [160, 210, 260]])
    es = E.run_ensemble(w, E.EnsembleParams.from_params(p0, 2), A, manage=False, forced_cuts=fc)
    assert list(es.n_cuts) == [2, 3]
    assert list(es.cut_day[0, :2]) == [150, 197] and list(es.cut_day[1, :3]) == [160, 210, 260]


def test_trace_matches_scalar(years, p0):
    w = years[2003]
    tr = []
    M.run(w, p0, A, trace=tr)
    tr = np.array(tr)
    _, et = E.run_ensemble(w, E.EnsembleParams.from_params(p0, 1), A, trace=True)
    for col, key in enumerate(("biomasse", "H", "omd", "W", "ST", "n_cuts", "lai"), start=1):
        np.testing.assert_allclose(et[key][:, 0], tr[:, col], rtol=RTOL, atol=1e-9, err_msg=key)


def test_to_states_roundtrip(years, p0):
    w = years[2012]
    s = M.run(w, p0, A)
    es = E.run_ensemble(w, E.EnsembleParams.from_params(p0, 2), A)
    back = es.to_states()[1]
    assert [c.day for c in back.cuts] == [c.day for c in s.cuts]
    np.testing.assert_allclose([c.netto_kg for c in back.cuts], [c.netto_kg for c in s.cuts], rtol=RTOL)


def test_gro_factor_one_is_neutral(years, p0):
    w = years[2009]
    a = E.run_ensemble(w, E.EnsembleParams.from_params(p0, 2), A)
    b = E.run_ensemble(w, E.EnsembleParams.from_params(p0, 2), A, gro_factor=np.ones(2))
    np.testing.assert_allclose(a.season_total(), b.season_total(), rtol=0)


def test_perturbations_mean_preserving(p0):
    rng = np.random.default_rng(0)
    ep = E.perturb_params(p0, 20000, rng, {"NI": 0.2})
    assert abs(ep.NI.mean() / p0.NI - 1) < 0.01
    assert np.all(ep.RUEmax == p0.RUEmax)
    es = E.perturb_state(E.EnsembleState.initial(ep), rng, 0.3)
    assert abs(es.GV.mean() / 400.0 - 1) < 0.02
