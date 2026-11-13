"""Bausteine des Zwillingsexperiments (heu/zwilling.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heu import zwilling as Z  # noqa: E402


def test_crps_point_ensemble_is_abs_error():
    assert abs(Z.crps_ensemble(np.full(10, 3.0), 5.0) - 2.0) < 1e-12


def test_crps_prefers_centered_ensemble():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000)
    assert Z.crps_ensemble(x, 0.0) < Z.crps_ensemble(x + 2.0, 0.0)


def test_synthetic_lai_respects_clouds_and_noise():
    lai = np.full(365, 3.0)
    obs = Z.synthetische_lai(lai, np.random.default_rng(1), revisit=5, p_bewoelkt=0.5)
    n_ueberfluege = len(range(60, 300, 5))
    assert 0.3 * n_ueberfluege < len(obs.tag) < 0.7 * n_ueberfluege
    assert np.all(obs.clear >= 0.8) and np.all(obs.sigma >= 0.75)
    assert abs(obs.lai.mean() - 3.0) < 3 * obs.sigma.mean() / np.sqrt(len(obs.tag))


def test_schnitt_ensemble_shift_bounds():
    t = Z.schnitt_ensemble([150, 200], 500, np.random.default_rng(2), 3)
    assert t.shape == (500, 2)
    assert t[:, 0].min() >= 147 and t[:, 0].max() <= 153
