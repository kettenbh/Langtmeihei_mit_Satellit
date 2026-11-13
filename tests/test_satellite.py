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
