"""Schnitterkennung: auch ohne erkannte Schnitte muss eine Tabelle mit allen Spalten zurückkommen."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heu import satellite as SAT  # noqa: E402

COLS = SAT.SCHNITT_SPALTEN


def _s2(ndvi, start="2024-04-01", step=5, lai=None):
    idx = pd.date_range(start, periods=len(ndvi), freq=f"{step}D")
    df = pd.DataFrame({"ndvi": ndvi, "clear": np.ones(len(ndvi))}, index=idx)
    if lai is not None:
        df["lai"] = lai
    return df


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


def test_cut_spread_over_several_images():
    """Mähen, Heu liegt, Abfahren: der Einbruch verteilt sich über drei Aufnahmen mit kleinen Schritten."""
    nd = [0.80, 0.84, 0.84, 0.82, 0.78, 0.74, 0.72, 0.78, 0.83, 0.86]
    lai = [2.5, 3.0, 3.2, 2.8, 2.2, 1.8, 1.5, 2.0, 2.6, 3.1]
    cuts = SAT.detect_cuts(_s2(nd, step=3, lai=lai))
    assert len(cuts) == 1 and cuts["kriterium"].iloc[0] == "LAI"
    assert cuts["frueh"].iloc[0] <= pd.Timestamp("2024-04-10") <= cuts["spaet"].iloc[0] + pd.Timedelta(days=6)


def test_haze_dip_with_immediate_rebound_is_ignored():
    nd = [0.85, 0.86, 0.87, 0.55, 0.87, 0.86]
    lai = [3.0, 3.2, 3.3, 1.5, 3.4, 3.3]
    assert SAT.detect_cuts(_s2(nd, lai=lai)).empty


def test_arable_year_classified_and_skipped():
    """Getreide: Abreife, Stoppel (NDVI < 0,3), Zwischenfrucht. Kein Grünland, also keine Schnitte."""
    nd = [0.6, 0.8, 0.85, 0.85, 0.8, 0.7, 0.55, 0.4, 0.3, 0.18, 0.17, 0.2, 0.45, 0.8]
    s2 = _s2(nd, start="2025-04-10", step=10)
    assert SAT.nutzung_je_jahr(s2) == {2025: SAT.NUTZUNG_ACKER}
    assert SAT.detect_cuts(s2).empty


def test_single_dark_image_does_not_make_arable():
    nd = [0.8, 0.85, 0.15, 0.86, 0.84, 0.8]
    assert SAT.nutzung_je_jahr(_s2(nd)) == {2024: SAT.NUTZUNG_GRUENLAND}
