"""Satellitenbild über die Process API: Anfrage richtig aufgebaut (CDSE gemockt)."""
import datetime as dt
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heu import satellite as SAT  # noqa: E402

FELD = {"id": "t", "geometry": {"type": "Polygon", "coordinates": [[[12.727, 48.101], [12.730, 48.101],
                                                                    [12.730, 48.104], [12.727, 48.101]]]}}


def test_anfrage(monkeypatch):
    gesendet = {}
    monkeypatch.setattr(SAT.CDSE, "process", lambda self, body: gesendet.setdefault("b", body) and b"PNG")
    png, bb = SAT.satellitenbild(FELD, dt.date(2026, 6, 20), "ndvi", fenster_tage=20, creds=("a", "b"))
    b = gesendet["b"]
    assert png == b"PNG" and bb[0] < 12.727 and bb[3] > 48.104
    assert b["input"]["data"][0]["dataFilter"]["timeRange"]["from"].startswith("2026-05-31")
    assert 30 < b["output"]["width"] < 80 and "SCL" in b["evalscript"]


def test_bild_datum():
    idx = pd.to_datetime(["2026-06-01", "2026-06-10", "2026-06-18", "2026-06-25"])
    s2 = pd.DataFrame({"clear": [1.0, 1.0, 0.3, 1.0]}, index=idx)
    assert SAT.bild_datum(s2, dt.date(2026, 6, 20)) == dt.date(2026, 6, 10)
    assert SAT.bild_datum(s2, dt.date(2026, 5, 1)) is None
