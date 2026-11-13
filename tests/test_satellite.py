"""Schnitterkennung: auch ohne erkannte Schnitte muss eine Tabelle mit allen Spalten zurückkommen."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heu import satellite as SAT  # noqa: E402

COLS = ["datum", "frueh", "spaet", "ndvi_vor", "ndvi_nach", "unsicherheit_tage", "s1_bestaetigt"]


def _s2(ndvi, start="2024-04-01", step=5):
    idx = pd.date_range(start, periods=len(ndvi), freq=f"{step}D")
    return pd.DataFrame({"ndvi": ndvi, "clear": np.ones(len(ndvi))}, index=idx)


def test_no_cuts_returns_all_columns():
    cuts = SAT.detect_cuts(_s2([0.7, 0.72, 0.75, 0.74, 0.76]))
    assert list(cuts.columns) == COLS and cuts.empty
    assert cuts[cuts["datum"].dt.year == 2024].empty   # so nutzt app.py die Tabelle


def test_empty_input_returns_all_columns():
    for s2 in (None, pd.DataFrame()):
        cuts = SAT.detect_cuts(s2)
        assert list(cuts.columns) == COLS and cuts.empty


def test_cut_detected():
    cuts = SAT.detect_cuts(_s2([0.7, 0.8, 0.82, 0.45, 0.55, 0.7]))
    assert len(cuts) == 1 and cuts["datum"].dt.year.iloc[0] == 2024


def test_clear_normalised_to_field_pixels():
    """Rohanteil max. 0,70 (Bounding Box) -> wolkenfreie Aufnahmen werden 1,0."""
    idx = pd.date_range("2024-04-01", periods=6, freq="5D")
    raw = pd.DataFrame({"ndvi": 0.7, "valid_px": [162, 162, 0, 81, 162, 161], "sample_px": 230}, index=idx)
    raw["clear"] = raw["valid_px"] / raw["sample_px"]
    out = SAT.normiere_clear(raw)
    np.testing.assert_allclose(out["clear"], [1, 1, 0, 0.5, 1, 161 / 162])
    np.testing.assert_allclose(SAT.normiere_clear(out)["clear"], out["clear"])   # idempotent


def test_clear_normalised_legacy_cache():
    idx = pd.date_range("2024-04-01", periods=5, freq="5D")
    old = pd.DataFrame({"ndvi": 0.7, "clear": [0.704, 0.704, 0.0, 0.352, 0.704]}, index=idx)
    np.testing.assert_allclose(SAT.normiere_clear(old)["clear"], [1, 1, 0, 0.5, 1])


def test_clear_not_scaled_without_enough_clear_scenes():
    idx = pd.date_range("2024-04-01", periods=4, freq="5D")
    d = pd.DataFrame({"ndvi": 0.7, "clear": [0.3, 0.1, 0.0, 0.2]}, index=idx)
    np.testing.assert_allclose(SAT.normiere_clear(d)["clear"], d["clear"])
