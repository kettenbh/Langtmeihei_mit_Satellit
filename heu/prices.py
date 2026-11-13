"""Heupreise: Einlesen einer selbst gepflegten Preisreihe und einfache Quantilprognose.

Es gibt keine frei maschinenlesbare Heupreis-Zeitreihe für Südostbayern. Deshalb liest das MVP
eine CSV-Datei (data/heupreise.csv), die aus LfL-Marktberichten, Raufutter-Notierungen der
Fachpresse oder eigenen Abschlüssen gefüllt wird. Format siehe data/heupreise_vorlage.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR

PRICE_FILE = DATA_DIR / "heupreise.csv"
MIN_MONATE = 24


def load_prices(segment: str | None = None) -> pd.DataFrame | None:
    if not PRICE_FILE.exists():
        return None
    df = pd.read_csv(PRICE_FILE, comment="#")
    if df.empty:
        return None
    df["monat"] = pd.to_datetime(df["monat"].astype(str).str.slice(0, 7) + "-01")
    if segment and "segment" in df:
        df = df[df["segment"] == segment]
    s = df.groupby("monat")["preis_eur_dt"].mean().sort_index()
    s = s.asfreq("MS").interpolate(limit=2)
    return s.to_frame("preis")


def price_forecast(prices: pd.DataFrame, horizon: int = 6, until: pd.Timestamp | None = None) -> pd.DataFrame | None:
    """Empirische Quantile der relativen Preisänderung je Horizont (Log-Renditen der Historie),
    angewandt auf den letzten bekannten Preis bis 'until'."""
    s = prices["preis"].dropna()
    if until is not None:
        s = s[s.index <= until]
    if len(s) < MIN_MONATE:
        return None
    logp = np.log(s)
    last_t, last_p = s.index[-1], s.iloc[-1]
    rows = []
    for h in range(1, horizon + 1):
        r = (logp.shift(-h) - logp).dropna()
        same_month = r[r.index.month == last_t.month]
        base = same_month if len(same_month) >= 5 else r
        qs = np.percentile(base, [10, 25, 50, 75, 90])
        rows.append({"monat": last_t + pd.DateOffset(months=h),
                     **{f"q{q}": last_p * np.exp(v) for q, v in zip((10, 25, 50, 75, 90), qs)}})
    return pd.DataFrame(rows).set_index("monat")
